import os
import random

import numpy as np
import torch


def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str, default="./llama-7b/", help="Base model path")
    parser.add_argument("--output_dir", type=str, default="./ckpt/", help="Output directory")
    return parser


def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="", help="Data directory")
    parser.add_argument("--tasks", type=str, default="seqrec", help="Downstream tasks")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".index.json", help="Item index file")

    parser.add_argument(
        "--max_his_len",
        type=int,
        default=20,
        help="Maximum number of items in the history sequence; -1 means no limit",
    )
    parser.add_argument("--add_prefix", action="store_true", default=False, help="Add position prefixes to history")
    parser.add_argument("--his_sep", type=str, default=", ", help="Separator used for history items")
    parser.add_argument("--only_train_response", action="store_true", default=True, help="Train only on responses")

    parser.add_argument(
        "--train_prompt_sample_num",
        type=str,
        default="1",
        help="Number of prompt samples for each task",
    )
    parser.add_argument(
        "--train_data_sample_num",
        type=str,
        default="0",
        help="Number of training samples for each task",
    )

    parser.add_argument("--max_seq_len", type=int, default=-1)
    parser.add_argument("--shift_flag", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser


def parse_train_args(parser):
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer name")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        help="Comma-separated target modules",
    )
    parser.add_argument(
        "--lora_modules_to_save",
        type=str,
        default="embed_tokens,lm_head",
        help="Comma-separated modules to save",
    )

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1309)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z2_bf16.json")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default="TK-Rec", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="default")
    return parser


def parse_test_args(parser):
    parser.add_argument("--ckpt_path", type=str, default="", help="Checkpoint path")
    parser.add_argument("--filter_items", action="store_true", default=False, help="Filter illegal items")
    parser.add_argument("--results_file", type=str, default="./results/test-ddp.json", help="Result output path")
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument(
        "--use_final_model",
        action="store_true",
        default=False,
        help="Evaluate all blocks from the final checkpoint plus per-block modules_to_save subsets",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
        help="Comma-separated evaluation metrics",
    )
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
