"""Model-only checkpoints for this isolated AutoDL run (DeepSpeed ZeRO-2).

Keep the original validation-based choice, but reload the serialized module
weights directly because no DeepSpeed optimizer checkpoint is written.
DynamicTrainer still restores the matching selection/controller state.
"""
import functools
import json
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file


def install_model_only_checkpoint_support(trainer_class):
    original_init = trainer_class.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.args.save_only_model = True
        print('MODEL_ONLY_CHECKPOINTS_ENABLED save_only_model=True '
              f'load_best_model_at_end={self.args.load_best_model_at_end}', flush=True)

    def load_best(self):
        if not self.state.best_model_checkpoint:
            raise RuntimeError('No validation-selected best model was recorded')
        if self.is_deepspeed_enabled and self.model_wrapped.zero_optimization_stage() != 2:
            raise RuntimeError('Model-only best reload has only been audited for ZeRO-2')
        folder = Path(self.state.best_model_checkpoint)
        state = load_file(str(folder / 'model.safetensors'), device='cpu')
        self.model.load_state_dict(state, strict=True)
        restored = self.model.state_dict()
        mismatches = [name for name, value in state.items()
                      if not torch.equal(restored[name].detach().cpu(), value)]
        if mismatches:
            raise RuntimeError('Best-model weights failed exact reload: ' + str(mismatches[:5]))
        del restored, state
        if dist.is_initialized():
            dist.barrier()
        if self.args.should_save:
            report = dict(best_model_checkpoint=str(folder), best_metric=self.state.best_metric,
                          weights_reload_exactly=True, save_only_model=True,
                          optimizer_checkpoint_saved=False,
                          checkpoint_selection='minimum validation loss, original evaluation cadence')
            Path(self.args.output_dir, 'model_only_best_reload.json').write_text(json.dumps(report, indent=2))
        print(f'MODEL_ONLY_BEST_RELOADED {folder} metric={self.state.best_metric}', flush=True)

    trainer_class.__init__ = initialize
    trainer_class._load_best_model = load_best
