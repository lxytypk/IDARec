"""Configuration and command construction shared by the launcher and pipeline."""

from decimal import Decimal
import json
import math
from pathlib import Path


FULL_LORA_PARAMETERS = 5_636_096
FULL_ATTENTION_PARAMETERS = 1_703_936
FULL_MLP_PARAMETERS = 3_932_160
FULL_ATTENTION_SHARE = FULL_ATTENTION_PARAMETERS / FULL_LORA_PARAMETERS


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_config(config):
    """Preserve partial-family budgets and the full-budget all-matrix behavior."""
    config = dict(config)
    if config.get("dataset", "Books_cl_ori") != "Books_cl_ori":
        raise ValueError("This launcher uses the Books_cl_ori experiment configuration")
    if config.get("seed", 42) != 42:
        raise ValueError("This experiment configuration uses seed 42")
    budget = float(config["budget_ratio"])
    share = float(config.get("attention_share", 0.4))
    learning_rate = float(config["actual_initial_lr"])
    if not 0 < budget <= 1:
        raise ValueError("budget_ratio must be in (0, 1]")
    if not 0 < share < 1:
        raise ValueError("attention_share must be in (0, 1)")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("actual_initial_lr must be finite and positive")
    gpus = config.get("gpus")
    if (not isinstance(gpus, list) or len(gpus) != 2
            or any(not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0 for gpu in gpus)
            or len(set(gpus)) != 2):
        raise ValueError("gpus must contain two different nonnegative GPU indices")
    interval = int(config.get("reselect_steps", 100))
    port = int(config.get("port", 53412))
    if interval <= 0 or not 1024 < port < 65525:
        raise ValueError("Use a positive reselect_steps value and a port between 1025 and 65524")
    total = math.floor(budget * FULL_LORA_PARAMETERS)
    if budget == 1:
        config["requested_attention_share"] = share
        share = FULL_ATTENTION_SHARE
        attention = FULL_ATTENTION_PARAMETERS
    else:
        attention = math.floor(total * share)
    config.update(dataset="Books_cl_ori", seed=42, budget_ratio=budget, attention_share=share,
                  actual_initial_lr=learning_rate, reselect_steps=interval, port=port, gpus=list(gpus),
                  all_AB_update=budget == 1, budget_parameters=total,
                  attention_budget_parameters=attention, mlp_budget_parameters=total - attention)
    return config


def build_training_arguments(config, run_directory):
    root = Path(run_directory)
    command_learning_rate = str(Decimal(str(config["actual_initial_lr"])) * 10)
    return [
        "--base_model", str(root / "inputs/base_model"),
        "--seed", str(config["seed"]), "--dataset", config["dataset"],
        "--data_path", str(root / "datasets"), "--learning_rate", command_learning_rate,
        "--epochs", "4", "--tasks", "seqrec", "--train_prompt_sample_num", "1",
        "--train_data_sample_num", "0", "--index_file", ".index.epoch20000_edim32_beta0.0.json",
        "--temperature", ".8", "--max_seq_len", "-1", "--max_his_len", "20",
        "--only_train_response", "--num_blocks", "5", "--lora_r", "8",
        "--lora_alpha", "16", "--lora_dropout", ".05",
        "--lora_target_modules", "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        "--lora_modules_to_save", "embed_tokens,lm_head", "--shift_flag", "lora_kldiv_latest",
        "--deepspeed", str(root / "train_code/ds_split_budget.json"),
        "--per_device_batch_size", "8", "--gradient_accumulation_steps", "2",
        "--bf16", "--optim", "adamw_torch", "--weight_decay", ".01",
        "--lr_scheduler_type", "cosine", "--warmup_ratio", ".01",
        "--save_and_eval_strategy", "epoch", "--probe_batches", "8", "--probe_batch_size", "1",
        "--probe_ratio", ".1", "--probe_seed", str(config["seed"]),
        "--budget_ratio", repr(config["budget_ratio"]),
        "--attention_budget_share", repr(config["attention_share"]),
        "--score_mode", "lora_delta_taylor", "--selection_score_mode", "gradient_per_parameter",
        "--replay_ratio", "0", "--history_samples", "256", "--history_seed", str(config["seed"]),
        "--reselect_steps", str(config["reselect_steps"]),
        "--regularization_mode", "none", "--regularization_weight", "0",
    ]


def build_training_command(config, run_directory, block, task, python):
    if block not in (1, 2, 3, 4) or task not in ("score", "prepare", "train"):
        raise ValueError("Use block 1-4 and a score, prepare, or train task")
    root = Path(run_directory)
    command = [python]
    if task == "train":
        command += ["-m", "torch.distributed.run", "--nproc_per_node=2",
                    "--master_port=" + str(config["port"] + block)]
    command += [str(root / "train_entry.py"), task, "--root", str(root),
                "--source_checkpoint", str(root / "ckpt" / ("block_" + str(block - 1))),
                "--block_id", str(block)]
    return command + build_training_arguments(config, root)


def build_evaluation_command(config, run_directory, result, python):
    root = Path(run_directory)
    return [
        python, "-m", "torch.distributed.run", "--nproc_per_node=2",
        "--master_port=" + str(config["port"] + 10), str(root / "eval_code/continual_test.py"),
        "--ckpt_path", str(root / "evaluation_checkpoints"),
        "--base_model", str(root / "inputs/base_model"), "--seed", str(config["seed"]),
        "--dataset", config["dataset"], "--data_path", str(root / "datasets"),
        "--results_file", str(result), "--test_batch_size", "1", "--num_beams", "10",
        "--index_file", ".index.epoch20000_edim32_beta0.0.json",
        "--shift_flag", "lora_kldiv_latest", "--num_blocks", "5",
        "--test_all_blocks", "--use_final_model", "--skip_block0",
        "--lora_r", "8", "--lora_alpha", "16", "--lora_dropout", ".05",
        "--lora_target_modules", "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        "--lora_modules_to_save", "embed_tokens,lm_head",
    ]
