"""LoRA causal model emitting independent suppression probabilities per slot."""
import json
from pathlib import Path
import torch
from .lora_actor import LoRAActionActor


class CumulativeMaskActor(LoRAActionActor):
    def __init__(self, model_path, rank=8, max_length=4096):
        super().__init__(model_path, rank=rank, max_length=max_length)
        # Retain the shared save/restore count contract; these vocabulary IDs
        # are never used as action logits by the independent linear mask head.
        digits = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)]
        if any(len(ids) != 1 for ids in digits):
            raise ValueError('Expected one token per decimal slot ID')
        self.action_tokens = [ids[0] for ids in digits]

    def initialize_mask_head(self):
        self.enable_index_policy_head(10)

    def inputs(self, conversations):
        if any(not isinstance(messages, list) for messages in conversations):
            raise ValueError('Pass exact teacher messages, not rewrapped strings')
        texts = [self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for messages in conversations]
        encoded = self.tokenizer(texts, padding=True, return_tensors='pt')
        if encoded.input_ids.shape[1] > self.max_length:
            raise ValueError('Cumulative observation exceeds the untruncated context budget')
        return encoded.to(self.model.device)

    def probabilities(self, messages, count):
        if not 0 <= count <= 10 or not hasattr(self.model, 'index_policy_head'):
            raise ValueError('Initialize or restore the ten-slot mask head')
        self.model.eval()
        with torch.no_grad():
            logits = self.policy_logits([messages])[0, :count]
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite mask logits')
            return logits.sigmoid().cpu().numpy()

    def save(self, path):
        super().save(path)
        (Path(path)/'cumulative_actor_config.json').write_text(json.dumps(dict(
            interface='independent_binary_suppression_v1', slots=10,
            observation='P36 original messages', threshold=.5))+'\n')

    def load(self, path, trainable=False):
        config = json.loads((Path(path)/'cumulative_actor_config.json').read_text())
        if config['interface'] != 'independent_binary_suppression_v1' or config['slots'] != 10:
            raise ValueError('Not a compatible cumulative-mask checkpoint')
        super().load(path, trainable=trainable)
        if len(self.action_tokens) != 10:
            raise ValueError('Cumulative checkpoint action count mismatch')
