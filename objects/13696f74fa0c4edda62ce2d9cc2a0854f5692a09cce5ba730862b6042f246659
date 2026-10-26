"""Optional training-only relative L2 anchor to the preceding saved LoRA pair."""
import torch


def relative_ab_anchor(module, previous_index):
    if previous_index < 0 or len(module.A_matrices) <= previous_index:
        raise RuntimeError('Missing preceding adapter for anchor regularization')
    a0=module.A_matrices[previous_index].detach().float()
    b0=module.B_matrices[previous_index].detach().float()
    denominator=(a0.square().sum()+b0.square().sum()).clamp_min(1e-12)
    return ((module.A.float()-a0).square().sum()+
            (module.B.float()-b0).square().sum()) / denominator
