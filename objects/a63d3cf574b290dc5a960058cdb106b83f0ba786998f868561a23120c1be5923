import os
from torch.utils.data import Dataset
from collections import defaultdict
import json
from prompt import sft_prompt, all_prompt
import numpy as np


class ContinualSeqRecDataset_v2(Dataset):
    """
    Block-based continual sequential-recommendation dataset for the public release.
    """
    
    def __init__(self, args, mode="train", 
                 prompt_sample_num=1, prompt_id=0, sample_num=-1,
                 current_block=0, num_blocks=5):
        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.current_block = current_block
        self.num_blocks = num_blocks
        
        # Data processing parameters
        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.add_prefix = args.add_prefix
        self.debug = args.debug
        self.prompt = all_prompt["seqrec"][0]

        self.new_tokens = None
        self.all_items = None
        self.allowed_tokens = None
        
        # Load and process data
        self._load_data()
        self._process_data_by_blocks()
        
        # Process data based on mode and current block
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        """Load block-based interaction data and index file (new format with previous_block/current_block)"""
        self.block_data = {}
        for block_id in range(self.num_blocks):
            block_file = os.path.join(self.data_path, f"{self.dataset}.all.block_{block_id}.json")
            if os.path.exists(block_file):
                with open(block_file, 'r') as f:
                    self.block_data[block_id] = json.load(f)
            else:
                print(f"Warning: Block file {block_file} not found")
                self.block_data[block_id] = {}
        self.index_file = self.args.index_file
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def _process_data_by_blocks(self):
        """Process interaction data by blocks and create user sequences."""
        self.block_user_sequences = {i: defaultdict(list) for i in range(self.num_blocks)}
        for block_id in range(self.num_blocks):
            block_data = self.block_data[block_id]
            for user_id, data in block_data.items():
                user_id = str(user_id)
                self.block_user_sequences[block_id][user_id] = {
                    "previous_block": data["previous_block"],
                    "current_block": data["current_block"],
                }
        self._remap_block_items()

    def _remap_block_items(self):
        """Map raw item ids to semantic token strings."""
        self.remapped_block_user_sequences = {i: defaultdict(list) for i in range(self.num_blocks)}
        for block_id in range(self.num_blocks):
            for uid, data in self.block_user_sequences[block_id].items():
                previous_block = data["previous_block"]
                current_block = data["current_block"]
                remapped_previous_block = ["".join(self.indices[str(i)]) for i in previous_block]
                remapped_current_block = ["".join(self.indices[str(i)]) for i in current_block]
                self.remapped_block_user_sequences[block_id][uid] = {
                    "previous_block": remapped_previous_block,
                    "current_block": remapped_current_block,
                }

    def _format_history(self, history):
        """Format an interaction history into the prompt string."""
        if self.max_his_len > 0:
            history = history[-self.max_his_len:]
        if self.add_prefix:
            history = [f"{k + 1}. {item_idx}" for k, item_idx in enumerate(history)]
        return self.his_sep.join(history)

    def _make_example(self, label, history):
        return {
            "item": label,
            "inters": self._format_history(history),
        }

    def _debug_sample(self, inter_data):
        if self.debug and len(inter_data) > 100:
            sample_idx = np.random.choice(len(inter_data), 100, replace=False)
            inter_data = np.array(inter_data, dtype=object)[sample_idx].tolist()
        return inter_data

    def _count_total_users(self):
        return max(1, len(self.remapped_block_user_sequences[self.current_block]))

    def _process_train_data(self):
        """Process training data for the current block."""
        inter_data = []
        sparse_user_count = 0
        for _, data in self.remapped_block_user_sequences[self.current_block].items():
            if len(data["current_block"]) < 3:
                sparse_user_count += 1
                continue

            prev = data["previous_block"]
            curr = data["current_block"][:-2]
            full_seq = prev + curr

            for idx in range(len(prev), len(full_seq)):
                inter_data.append(self._make_example(full_seq[idx], full_seq[:idx]))

        inter_data = self._debug_sample(inter_data)
        print(f"Train sparse user count / total user count: {sparse_user_count / self._count_total_users()}")
        return inter_data

    def _process_valid_data(self):
        """Process validation data for the current block."""
        inter_data = []
        sparse_user_count = 0
        for _, data in self.remapped_block_user_sequences[self.current_block].items():
            if len(data["current_block"]) < 3:
                sparse_user_count += 1
                continue
            prev = data["previous_block"]
            curr = data["current_block"]
            full_seq = prev + curr

            inter_data.append(self._make_example(full_seq[-2], full_seq[:-2]))

        inter_data = self._debug_sample(inter_data)
        print(f"Valid sparse user count / total user count: {sparse_user_count / self._count_total_users()}")
        return inter_data

    def _process_test_data(self):
        """Process test data for the current block."""
        inter_data = []
        sparse_user_count = 0
        for _, data in self.remapped_block_user_sequences[self.current_block].items():
            if len(data["current_block"]) < 3:
                sparse_user_count += 1
                continue
            prev = data["previous_block"]
            curr = data["current_block"]
            full_seq = prev + curr
            inter_data.append(self._make_example(full_seq[-1], full_seq[:-1]))

        print(f"Test sparse user count / total user count: {sparse_user_count / self._count_total_users()}")

        return inter_data

    def set_current_block(self, block_id: int):
        """Switch to a different chronological block and rebuild cached examples."""
        assert 0 <= block_id < self.num_blocks, f"Block ID {block_id} out of range [0, {self.num_blocks})"
        self.current_block = block_id
        
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
            self._construct_valid_text()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()

    def set_prompt(self, prompt_id):
        if prompt_id != 0:
            raise ValueError("Only prompt_id=0 is supported in the public release.")

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data) * self.prompt_sample_num
        elif self.mode == 'valid':
            return len(self.valid_text_data)
        elif self.mode == 'test':
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def _construct_valid_text(self):
        """Construct validation examples for the fixed public prompt."""
        self.valid_text_data = []
        self.prompt_sample_num = 1
        
        for i in range(len(self.inter_data)):
            d = self.inter_data[i]
            input, output = self._get_text_data(d)
            self.valid_text_data.append({"input_ids": input, "labels": output})

    def _get_text_data(self, data):
        """Render the canonical prompt and target for one example."""
        instruction = self.prompt["instruction"].format(**data)
        response = self.prompt["response"].format(**data)

        input = sft_prompt.format(instruction=instruction, response="")
        output = sft_prompt.format(instruction=instruction, response=response)

        if self.mode == 'test':
            return input, response

        return input, output

    def __getitem__(self, index):
        if self.mode == 'valid':
            return self.valid_text_data[index]

        idx = index // self.prompt_sample_num
        d = self.inter_data[idx]
        input, output = self._get_text_data(d)

        return dict(input_ids=input, labels=output)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens
    

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):

        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][1]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = tokenizer("### Response:")["input_ids"][1:]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    return list(self.allowed_tokens[i]) 

        return prefix_allowed_tokens_fn


def load_continual_datasets_v2(args, current_block=0, num_blocks=5):
    """
    Load continual learning datasets for a specific block.
    
    Args:
        args: Arguments object
        current_block: Current block to train on
        num_blocks: Total number of blocks
    
    Returns:
        train_data, valid_data: ContinualSeqRecDataset objects
    """
    tasks = args.tasks.split(",")
    if "seqrec" not in tasks:
        raise NotImplementedError("Continual learning currently only supports seqrec task")
    
    train_prompt_sample_num = [int(_) for _ in args.train_prompt_sample_num.split(",")]
    train_data_sample_num = [int(_) for _ in args.train_data_sample_num.split(",")]
    
    seqrec_idx = tasks.index("seqrec")
    prompt_sample_num = train_prompt_sample_num[seqrec_idx]
    data_sample_num = train_data_sample_num[seqrec_idx]
    
    train_data = ContinualSeqRecDataset_v2(
        args, mode="train", 
        prompt_sample_num=prompt_sample_num, 
        sample_num=data_sample_num,
        current_block=current_block,
        num_blocks=num_blocks
    )
    
    valid_data = ContinualSeqRecDataset_v2(
        args, mode="valid",
        current_block=current_block,
        num_blocks=num_blocks
    )
    
    return train_data, valid_data 
