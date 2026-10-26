import math

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.split("Response:")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ","") for _ in predictions]

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError

    return res


def ndcg_k(topk_results, k):

    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit


# Distributed evaluation utilities: no padding, deterministic beam search,
# and one metric aggregation after each rank finishes its local examples.
import datetime
import json
import pathlib
from torch.utils.data import Sampler
from transformers import GenerationConfig

METRICS = ['hit@1', 'hit@5', 'hit@10', 'ndcg@5', 'ndcg@10']
GENERATION = dict(do_sample=False, num_beams=10, num_return_sequences=10,
                  max_new_tokens=10, early_stopping=True)

class ExactSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, **kwargs):
        self.n, self.world, self.rank = len(dataset), num_replicas, rank
    def __iter__(self):
        return iter(range(self.rank, self.n, self.world))
    def __len__(self):
        return len(range(self.rank, self.n, self.world))

def pin_generation(model, args, block):
    config = GenerationConfig.from_model_config(model.config)
    config.pad_token_id = model.config.pad_token_id
    config.eos_token_id = model.config.eos_token_id
    for key, value in GENERATION.items():
        setattr(config, key, value)
    config.output_scores = True
    config.return_dict_in_generate = True
    model.generation_config = config

def coverage_metadata(dataset, args, block, tokenizer):
    raw = dataset.block_data[block]
    users = [str(key) for key, row in raw.items() if len(row['current_block']) >= 3]
    return dict(original_users=len(raw), eligible_users=len(users), row_to_user=users)

def gather_complete(metrics_results, total, seen_indices, coverage, block, world):
    import torch.distributed as dist
    shards = [None] * world
    dist.all_gather_object(shards, dict(metrics=metrics_results, count=total))
    total = sum(row['count'] for row in shards)
    values = {metric: sum(row['metrics'].get(metric, 0.0) for row in shards) / total
              for metric in METRICS}
    return values, total

def save_progress(args, rows):
    done = sorted(int(key.split('_')[1]) for key in rows)
    result = dict(generation_mode='beam-search', evaluated_blocks=[1, 2, 3, 4],
                  skipped_blocks=[0], stage_restoration='historical_stage_state',
                  completed_blocks=done, status='completed' if done == [1, 2, 3, 4] else 'incomplete',
                  generation_configuration=GENERATION, all_block_results=rows)
    if done == [1, 2, 3, 4]:
        result['macro_average'] = {metric: sum(rows[f'block_{block}']['all_prompt_results'][0][metric]
                                              for block in range(1, 5)) / 4 for metric in METRICS}
    path = pathlib.Path(args.results_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
