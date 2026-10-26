"""Family-constrained LoRA matrix budget selection.

The total LoRA A/B parameter budget is split between the four attention
projections and the three MLP projections.  Candidates still compete by the
existing score within their family, so only the budget routing changes.
"""

import math


ATTENTION_PROJECTIONS = frozenset(("q_proj", "k_proj", "v_proj", "o_proj"))
MLP_PROJECTIONS = frozenset(("gate_proj", "up_proj", "down_proj"))


def projection_family(projection):
    if projection in ATTENTION_PROJECTIONS:
        return "attention"
    if projection in MLP_PROJECTIONS:
        return "mlp"
    raise ValueError("Unknown LoRA projection: %s" % projection)


def split_budget_limits(total_budget, attention_share):
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if not 0.0 < attention_share < 1.0:
        raise ValueError("attention_budget_share must be in (0, 1)")
    attention = int(math.floor(total_budget * attention_share))
    return {"attention": attention, "mlp": int(total_budget - attention)}


def greedy_selection_split(score_rows, total_budget, attention_share):
    """Greedily select score-sorted rows inside fixed family budget ceilings."""
    limits = split_budget_limits(total_budget, attention_share)
    used = {"attention": 0, "mlp": 0}
    selected = []
    for row in score_rows:
        family = projection_family(row["projection"])
        cost = int(row["lora_cost"])
        if used[family] + cost <= limits[family]:
            selected.append(row["matrix"])
            used[family] += cost
    return selected, used, limits


def family_counts(score_rows, selected):
    selected = set(selected)
    counts = {"attention": 0, "mlp": 0}
    for row in score_rows:
        if row["matrix"] in selected:
            counts[projection_family(row["projection"])] += 1
    return counts
