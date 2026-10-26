#!/usr/bin/env python3
"""Summarize the two PESO final-model all-block evaluations."""

import json
import os
from statistics import mean


BASE = "/opt/idarec/runs/taylor_lora/runs"
RUNS = {
    "attention60_mlp40": "attn60_mlp40_budget50",
    "attention40_mlp60": "attn40_mlp60_budget50",
}
RESULT_NAME = "peso_final_model_all_blocks_sampling.json"
OUTPUT = os.path.join(BASE, "peso_final_model_split_budget50_comparison.json")


def summarize(run_name):
    path = os.path.join(BASE, run_name, "evaluation", RESULT_NAME)
    payload = json.load(open(path, encoding="utf-8"))
    blocks = {}
    for block in range(5):
        row = payload["all_block_results"][f"block_{block}"]
        blocks[str(block)] = row["all_prompt_results"][0]

    def macro(block_ids):
        names = blocks[str(block_ids[0])]
        return {
            metric: mean(blocks[str(block)][metric] for block in block_ids)
            for metric in names
        }

    return {
        "source": path,
        "eval_protocol": payload.get("eval_protocol"),
        "blocks": blocks,
        "block0_to_block4_macro": macro(range(5)),
        "block1_to_block4_macro": macro(range(1, 5)),
    }


comparison = {name: summarize(run) for name, run in RUNS.items()}
left = comparison["attention60_mlp40"]["block1_to_block4_macro"]
right = comparison["attention40_mlp60"]["block1_to_block4_macro"]
summary = {
    "method": {
        "upstream": "https://github.com/hsyoo32/peso",
        "upstream_commit": "edbec5f68ddc83e39f22a041f1f2eaf1c4ad172e",
        "checkpoint_rule": "load final block_4 for every test block",
        "historical_state_rule": "restore block-specific saved LoRA matrices and modules_to_save",
        "test_rule": "test_all_blocks with use_final_model",
        "decode": "Llama-3.2-1B shipped BEAM_SAMPLE: do_sample=true, temperature=0.6, top_p=0.9, num_beams=10",
        "coverage_difference_from_upstream": "ExactDistributedEvalSampler covers every example; upstream DistributedSampler(drop_last=True) can omit one example from odd-sized blocks",
    },
    "comparison": comparison,
    "block1_to_block4_delta_40_60_minus_60_40": {
        metric: right[metric] - left[metric] for metric in left
    },
}
with open(OUTPUT, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
print(OUTPUT)
