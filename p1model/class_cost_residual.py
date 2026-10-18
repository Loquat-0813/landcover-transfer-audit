"""Prototype class-cost interaction residual; no trained/admitted model yet.

Shared pair MLPs accept variable leaf counts and commute with leaf permutations.
The own-score control has equal trainable count and informative inputs, but a
different width/input graph and effective capacity. Neither is CAT-Seg itself.
Zero final initialization preserves input logits, not post-training performance.
"""
import torch
from torch import nn,Tensor
import torch.nn.functional as F

ARMS=('own_score','pairwise_cost')

class ClassCostResidual(nn.Module):
    def __init__(self,arm:str):
        super().__init__()
        if arm not in ARMS:raise ValueError('explicit own_score or pairwise_cost arm required')
        self.arm=arm
        dim,width=(2,48) if arm=='own_score' else (4,32)
        self.network=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        assert sum(p.numel() for p in self.parameters())==193

    def correction(self,logits:Tensor,text:Tensor)->Tensor:
        if logits.ndim!=4 or text.ndim!=2 or logits.shape[1]!=text.shape[0]:
            raise ValueError('logits[B,L,H,W] and text[L,D] must share leaf axis')
        b,n,h,w=logits.shape
        if n<1:raise ValueError('empty leaf bank')
        if n==1:return torch.zeros_like(logits)
        t=F.normalize(text.detach(),dim=1)
        similarity=(t@t.T).to(dtype=logits.dtype)
        own=logits.unsqueeze(2).expand(b,n,n,h,w)
        semantic=similarity.view(1,n,n,1,1).expand(b,n,n,h,w)
        if self.arm=='own_score':inputs=torch.stack((own,semantic),dim=-1)
        else:
            other=logits.unsqueeze(1).expand(b,n,n,h,w)
            inputs=torch.stack((own,other,own-other,semantic),dim=-1)
        pair=self.network(inputs).squeeze(-1)
        mask=(~torch.eye(n,dtype=torch.bool,device=logits.device)).view(1,n,n,1,1)
        return pair.masked_fill(~mask,0).sum(dim=2)/(n-1)

    def forward(self,logits:Tensor,text:Tensor)->Tensor:
        return logits+self.correction(logits,text)
