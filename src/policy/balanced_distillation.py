"""Fit-only, globally normalized binary class weights (1 means suppress)."""


def class_weights(rows, fit_groups):
    if not rows or len({r['qid'] for r in rows}) != len(rows):
        raise ValueError('Unique nonempty fit observations required')
    if any(r['group'] not in fit_groups for r in rows):
        raise ValueError('Nonfit observations cannot determine class weights')
    masks = [r['mask'] for r in rows]
    if any(not 0 < len(m) <= 10 for m in masks):
        raise ValueError('Valid slots required')
    values = [v for m in masks for v in m]
    if any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError('Binary suppression labels required')
    suppress = sum(values)
    keep = len(values) - suppress
    if not keep or not suppress:
        raise ValueError('Both fit classes required')
    return dict(keep_slots=keep, suppress_slots=suppress,
                keep_weight=len(values)/(2*keep), suppress_weight=len(values)/(2*suppress))


def balanced_bce(logits, targets, weights):
    import torch
    import torch.nn.functional as F
    weight = torch.where(targets == 0, weights['keep_weight'], weights['suppress_weight'])
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weight)
