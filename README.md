# IDARec

Code for **Not All Modules Need Updating: Importance-Guided Dynamic Adaptation for Continual Learning in Generative Recommendation.**

## Setup

Run from this directory on Linux with Python 3.10+, CUDA, and two NVIDIA GPUs:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Use a PyTorch build compatible with your CUDA environment. FlashAttention must
match the installed PyTorch and CUDA versions. Transformers is pinned to 4.52.4;
a complete environment lock is not provided.

## Run

The provided launcher targets **Books with Llama-3.2-1B**. Prepare the base model
with its tokenizer and the corresponding Block-0 checkpoint locally; model
weights are not included in this repository.

The three datasets are included in `data/Instruments_cl_ori/`,
`data/Movies_and_TV_cl_ori/`, and `data/Books_cl_ori/`. Each contains interaction
splits for blocks 0–4, SID mappings, and item metadata. Metadata files
`.item.part001.json`, `.item.part002.json`, etc. are loaded together
automatically. Instruments uses the original PESO SID mapping; Movies and Books
use unique four-token SIDs. Dataset `block_0.json` files are included.

Copy the resource configuration:

```bash
cp resources.json resources.local.json
```

Edit `resources.local.json`: set `base_model` and `block0` to your local model
and checkpoint directories, and set `environment` to `.venv` if you created the
environment above. The default `data` path is `data/Books_cl_ori`.
`resources.local.json` is ignored by Git.

```bash
# Train the continual stages, then evaluate recommendations.
python run.py --config configs/budget40_lr1e4.json \
  --output runs/books_budget40 --launch
```

The default uses **40% of the full LoRA A/B parameter budget**, with **40% of
that budget assigned to Attention and 60% to MLP**. It uses seed 42, an initial
learning rate of `1e-4`, GPUs `[0, 1]`, and matrix reselection every 100 optimizer
steps. The pipeline continues from Block 0, trains blocks 1–4, and then evaluates
recommendations.

Use the other JSON files in `configs/` for 10%, 20%, 30%, 50%, 60%, 70%, 80%,
or 100% budgets. Partial budgets use `attention_share` to divide the budget;
the 100% configuration enables all LoRA A/B matrices. Learning rate,
reselection interval, GPU indices, and communication port are configurable.

To prepare a run without starting training:

```bash
python run.py --config configs/budget40_lr1e4.json \
  --output runs/books_budget40_prepared
```

Use a new output directory for each experiment. The launcher copies the current
`src/` into that directory, so source edits apply to newly prepared runs.
Training logs are written to `<output>/logs/launcher.log`; the launched pipeline
runs in the background.

## Acknowledgements

IDARec builds on [PESO](https://github.com/hsyoo32/peso) and uses its original
Instruments dataset. Movies and Books use the author-constructed temporal
datasets and unique SID mappings.
