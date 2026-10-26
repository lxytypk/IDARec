import datetime, os, torch, torch.distributed as dist
rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=45))
t=torch.tensor([rank+1.],device='cuda')
dist.all_reduce(t)
assert t.item()==3
print('NCCL_SUM_OK',rank,t.item(),flush=True)
dist.destroy_process_group()
