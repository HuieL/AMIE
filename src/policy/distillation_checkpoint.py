"""Atomic optimizer and RNG checkpoints for deterministic distillation replay."""
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, step, identity, losses):
    state=dict(identity=identity,step=step,losses=list(losses),
        parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
        optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(state,temporary);temporary.replace(path)


def restore_checkpoint(path, model, optimizer, identity):
    # Only task-owned tensor/primitive checkpoints, never arbitrary pickle objects.
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['identity']!=identity:raise ValueError('Training checkpoint protocol mismatch')
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if parameters.keys()!=state['parameters'].keys():raise ValueError('Trainable parameter interface changed')
    if any(parameters[n].shape!=v.shape or parameters[n].dtype!=v.dtype for n,v in state['parameters'].items()):
        raise ValueError('Trainable tensor interface changed')
    with torch.no_grad():
        for n,p in parameters.items():p.copy_(state['parameters'][n])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'])
    if state['cuda_rng']:
        if len(state['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA checkpoint device count changed')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step'],state['losses']
