#!/usr/bin/env python3
"""Create a non-destructive PESO final-model evaluation checkpoint layout."""

import argparse
import hashlib
import json
import os
import shutil

from safetensors.torch import load_file, save_file


TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_subset(source, destination):
    state = load_file(source, device="cpu")
    subset = {
        key: value
        for key, value in state.items()
        if ".modules_to_save.default." in key
    }
    if not subset:
        raise RuntimeError(f"No modules_to_save tensors found in {source}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".partial"
    save_file(subset, temporary)
    os.replace(temporary, destination)
    return len(subset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    args = parser.parse_args()

    run_root = os.path.abspath(args.run_root)
    source_root = os.path.join(run_root, "ckpt")
    final_root = os.path.join(run_root, "final_block4")
    eval_root = os.path.join(run_root, "peso_eval_ckpt")
    os.makedirs(eval_root, exist_ok=True)

    manifest = {
        "method": "PESO final-checkpoint layout",
        "source_run": run_root,
        "final_model_source": os.path.realpath(final_root),
        "historical_modules": {},
    }

    for filename in TOKENIZER_FILES:
        source = os.path.join(final_root, filename)
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        shutil.copy2(source, os.path.join(eval_root, filename))

    for block in range(4):
        source = os.path.join(source_root, f"block_{block}", "model.safetensors")
        destination = os.path.join(
            eval_root, f"block_{block}", "modules_to_save.safetensors"
        )
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        tensor_count = extract_subset(source, destination)
        manifest["historical_modules"][str(block)] = {
            "source": os.path.realpath(source),
            "source_sha256": sha256(source),
            "output": destination,
            "tensor_count": tensor_count,
            "output_sha256": sha256(destination),
        }

    final_link = os.path.join(eval_root, "block_4")
    if os.path.lexists(final_link):
        if os.path.islink(final_link) and os.path.realpath(final_link) == os.path.realpath(final_root):
            pass
        else:
            raise RuntimeError(f"Refusing to replace existing path: {final_link}")
    else:
        os.symlink(final_root, final_link)

    manifest_path = os.path.join(eval_root, "peso_eval_layout_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(manifest_path)


if __name__ == "__main__":
    main()
