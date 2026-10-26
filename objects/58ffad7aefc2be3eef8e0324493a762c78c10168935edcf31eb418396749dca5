import json,pathlib,sys,torch
import torch.nn.functional as F
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'train_code'))
from continual_lora import ContinualLoRALinear
from score_variants import module_score
torch.manual_seed(42)
module=ContinualLoRALinear(torch.nn.Linear(8,7,bias=False),2,4,.05,option='lora_kldiv_latest')
with torch.no_grad(): module.B.normal_(0,.2)
module.eval()
module.A.requires_grad_(False);module.B.requires_grad_(False)
sample_scores=[];gradients=[];deltas=[]
for i in range(3):
    x=torch.randn(3,8);target=torch.tensor([i,(i+1)%7,(i+2)%7])
    module.orig.weight.requires_grad_(True);module.zero_grad(set_to_none=True)
    F.cross_entropy(module(x)/.8,target).backward()
    g=module.orig.weight.grad.detach().clone()
    eff=(module.orig.weight.detach()+module.scaling*module.B@module.A).detach().requires_grad_(True)
    F.cross_entropy(F.linear(x,eff)/.8,target).backward()
    delta=float((g-eff.grad).abs().max())
    assert torch.allclose(g,eff.grad,atol=2e-7,rtol=2e-6),delta
    score=module_score(module,g,'lora_delta_taylor')
    expected=((module.scaling*(module.B.float()@module.A.float()))*g.float()).abs().sum(dtype=torch.float64)
    assert torch.equal(score,expected) and score.dtype==torch.float64
    gradients.append(g);sample_scores.append(float(score));deltas.append(delta)
cost=module.A.numel()+module.B.numel()
correct=sum(sample_scores)/cost
wrong=float(((module.scaling*module.B@module.A)*torch.stack(gradients).mean(0)).abs().sum(dtype=torch.float64))/cost
assert correct!=wrong
report=dict(gradient_identity='grad_orig_weight equals grad_effective_weight with dropout disabled',
            max_absolute_gradient_difference=max(deltas),per_sample_scores=sample_scores,
            pair_cost=cost,normalized_score=correct,mean_gradient_first_score=wrong,
            product_dtype='float32',sum_dtype='float64',samples=3,passed=True)
(ROOT/'audit').mkdir(exist_ok=True)
(ROOT/'audit/gradient_path.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
