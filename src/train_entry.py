import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'train_code'))
import matrix_dynamic50_split as experiment
# Synchronize the validation-selected path after rank 0 finishes serialization
# and checkpoint rotation. Trainer 4.52.4 checks path existence on each rank
# while only rank 0 creates it, so unsynchronized ranks can retain stale paths.
import torch.distributed as dist
from dynamic_matrix_control_split import DynamicTrainer
original_save_checkpoint = DynamicTrainer._save_checkpoint
def synchronized_save_checkpoint(self, *args, **kwargs):
    result = original_save_checkpoint(self, *args, **kwargs)
    if dist.is_initialized():
        dist.barrier()
        payload = [dict(best_model_checkpoint=self.state.best_model_checkpoint,
                        best_metric=self.state.best_metric,
                        best_global_step=self.state.best_global_step) if dist.get_rank()==0 else None]
        dist.broadcast_object_list(payload,src=0)
        for key,value in payload[0].items():setattr(self.state,key,value)
        path=pathlib.Path(self.state.best_model_checkpoint)
        assert (path/'model.safetensors').is_file(),str(path)
        loss_values=[x['eval_loss'] for x in self.state.log_history if 'eval_loss' in x]
        assert self.state.best_metric==min(loss_values)
    return result
DynamicTrainer._save_checkpoint=synchronized_save_checkpoint

experiment.main()
