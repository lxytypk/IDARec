"""Per-example scoring signals; selection unit, budgets, and optimizer stay unchanged."""
import torch
from replay import score_contribution

MODES = ('grad_weight_abs', 'grad_norm', 'lora_ab_grad_sq',
         'lora_delta_taylor', 'lora_ab_taylor', 'lora_ab_fisher')


def formula(mode):
    return {
        'grad_weight_abs': 'sum_i ||W * G_i||_1 / C',
        'grad_norm': 'mean_i ||G_i||_F / C',
        'lora_ab_grad_sq': '(||mean_i gA_i||_F^2 + ||mean_i gB_i||_F^2) / C',
        'lora_delta_taylor': 'sum_i ||((alpha/r) B A) * G_i||_1 / C',
        'lora_ab_taylor': 'sum_i (||A * gA_i||_1 + ||B * gB_i||_1) / C',
        'lora_ab_fisher': 'sum_i (||gA_i||_F^2 + ||gB_i||_F^2) / C; sum and mean give the same within-probe ranking',
    }[mode]


@torch.no_grad()
def module_score(module, gradient, mode):
    if mode in ('grad_weight_abs','grad_norm'):
        return score_contribution(module.orig.weight, gradient, mode)
    if mode == 'lora_ab_grad_sq':
        raise ValueError('Mean-then-square uses the dedicated signed-gradient accumulator')
    if module.training or module.dropout.training or module.merged:
        raise RuntimeError('Scoring must use unmerged eval-mode LoRA')
    g = gradient.detach().float()
    a,b = module.A.detach().float(),module.B.detach().float()
    scale=float(module.scaling)
    if mode == 'lora_delta_taylor':
        return ((scale * b.matmul(a)) * g).abs().sum(dtype=torch.float64)
    ga = scale * b.t().matmul(g)
    gb = scale * g.matmul(a.t())
    if mode == 'lora_ab_taylor':
        return (a * ga).abs().sum(dtype=torch.float64)+(b * gb).abs().sum(dtype=torch.float64)
    if mode == 'lora_ab_fisher':
        return ga.square().sum(dtype=torch.float64)+gb.square().sum(dtype=torch.float64)
    raise ValueError(mode)
