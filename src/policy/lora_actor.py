"""A real causal LLM with LoRA; a trainable linear policy head on top."""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def dequantize_fp8_linears(model):
    """Restore the checkpoint's block-quantized weights to trainable BF16 ops.

    This expands the *quantized values*, not the original pre-quantization
    model. Transformers 4.57's fine-grained FP8 Triton path is inference-only.
    Frozen and trained baselines both use this identical restored backbone.
    """
    from transformers.integrations.finegrained_fp8 import FP8Linear
    converted = 0
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, FP8Linear):
            continue
        rows, cols = layer.weight.shape
        br, bc = layer.block_size
        if rows % br or cols % bc:
            raise ValueError(f"Unsupported partial FP8 block in {name}")
        with torch.no_grad():
            weight = layer.weight.float().reshape(rows // br, br, cols // bc, bc)
            weight = (weight * layer.weight_scale_inv[:, None, :, None]).reshape(rows, cols)
            replacement = nn.Linear(cols, rows, bias=layer.bias is not None, device="meta")
            replacement.weight = nn.Parameter(weight.to(torch.bfloat16), requires_grad=False)
            if layer.bias is not None:
                replacement.bias = nn.Parameter(layer.bias.to(torch.bfloat16), requires_grad=False)
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, replacement)
        converted += 1
    if converted:
        if hasattr(model, "hf_quantizer"):
            del model.hf_quantizer
        model.is_quantized = False
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config
        torch.cuda.empty_cache()
    return converted


class LoRAActionActor:
    def __init__(self, model_path, rank=8, max_length=384):
        from peft import LoraConfig, get_peft_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, device_map={"": 0},
            dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        base.requires_grad_(False)
        self.converted_layers = dequantize_fp8_linears(base)
        self.model = get_peft_model(base, LoraConfig(
            task_type="CAUSAL_LM", r=rank, lora_alpha=2 * rank,
            lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        ))
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.max_length = max_length
        self.action_tokens = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in range(3)]
        if any(len(ids) != 1 for ids in self.action_tokens):
            raise ValueError("Action digits must each be a single tokenizer token")
        self.action_tokens = [ids[0] for ids in self.action_tokens]
        self.calls = 0

    def inputs(self, prompts):
        texts = [self.tokenizer.apply_chat_template([
            {"role": "system", "content": "You control an agent's persistent memory access index."},
            {"role": "user", "content": p},
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
        encoded = self.tokenizer(texts, padding=True, return_tensors="pt")
        if encoded.input_ids.shape[1] > self.max_length:
            raise ValueError(f"Prompt exceeds {self.max_length} tokens; do not silently truncate the action contract")
        return encoded.to(self.model.device)

    def logits(self, prompts):
        return self.model(**self.inputs(prompts), logits_to_keep=1).logits[:, -1, :].float()

    def enable_index_policy_head(self, n_actions):
        """Trainable linear action head, avoiding pretrained character priors.

        The causal decoder and its LoRA adapters still condition all actions on
        the index observation. No memory content is generated or modified.
        """
        if hasattr(self.model, 'index_policy_head'):
            raise ValueError('Policy head already initialized')
        head = nn.Linear(self.model.config.hidden_size, n_actions,
                         device=self.model.device, dtype=torch.float32)
        nn.init.normal_(head.weight, std=.01)
        nn.init.zeros_(head.bias)
        self.model.add_module('index_policy_head', head)

    def policy_logits(self, prompts):
        if not hasattr(self.model, 'index_policy_head'):
            return self.logits(prompts)[:, self.action_tokens]
        decoder = self.model.get_base_model().model
        hidden = decoder(**self.inputs(prompts)).last_hidden_state[:, -1, :]
        return self.model.index_policy_head(hidden.float())

    def act(self, prompt):
        self.model.eval()
        with torch.inference_mode():
            logits = self.policy_logits([prompt])[0]
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite policy logits")
        self.calls += 1
        return int(logits.argmax())

    def save(self, path):
        import json
        from pathlib import Path
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (Path(path)/'index_actor_config.json').write_text(json.dumps({
            'action_tokens': self.action_tokens, 'max_length': self.max_length,
            'head': 'linear' if hasattr(self.model, 'index_policy_head') else 'vocabulary'}))
        if hasattr(self.model, 'index_policy_head'):
            from safetensors.torch import save_file
            save_file({k: v.detach().contiguous().cpu() for k, v in self.model.index_policy_head.state_dict().items()},
                      str(Path(path)/'index_policy_head.safetensors'))

    def load(self, path, trainable=False):
        """Restore LoRA and the exact action interface on the same base model."""
        import json
        from pathlib import Path
        from safetensors.torch import load_file
        path = Path(path)
        config = json.loads((path/'index_actor_config.json').read_text())
        if hasattr(self.model, 'index_policy_head'):
            raise ValueError('Restore into a fresh actor, before head initialization')
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        saved_config = json.loads((path/'adapter_config.json').read_text())
        current = self.model.peft_config['default'].to_dict()
        for key in ('r', 'lora_alpha', 'lora_dropout', 'bias', 'use_rslora', 'use_dora'):
            if saved_config.get(key) != current.get(key):
                raise ValueError(f'Saved adapter configuration differs: {key}')
        if set(saved_config['target_modules']) != set(current['target_modules']):
            raise ValueError('Saved adapter target modules differ')
        weights = load_file(str(path/'adapter_model.safetensors'))
        existing = get_peft_model_state_dict(self.model, adapter_name='default')
        if set(weights) != set(existing) or any(weights[k].shape != existing[k].shape for k in weights):
            raise ValueError('Saved adapter tensor interface differs')
        set_peft_model_state_dict(self.model, weights, adapter_name='default')
        self.model.set_adapter('default')
        actual = get_peft_model_state_dict(self.model, adapter_name='default')
        if any(not torch.equal(weights[k].cpu(), actual[k].detach().cpu()) for k in weights):
            raise ValueError('Adapter restore changed saved weight values')
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad_(trainable)
        self.action_tokens = config['action_tokens']
        self.max_length = config['max_length']
        if config['head'] == 'linear':
            state = load_file(str(path/'index_policy_head.safetensors'))
            if state['weight'].shape[0] != len(self.action_tokens):
                raise ValueError('Saved action count differs from head')
            self.enable_index_policy_head(len(self.action_tokens))
            self.model.index_policy_head.load_state_dict(state)
            self.model.index_policy_head.requires_grad_(trainable)
        elif config['head'] != 'vocabulary':
            raise ValueError('Unknown saved policy head')
        self.model.eval()
