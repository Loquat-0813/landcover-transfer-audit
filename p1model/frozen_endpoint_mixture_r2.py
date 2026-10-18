"""Frozen KL/dense endpoint interpolation; exact gate-off return, no safety claim."""
import torch
from torch import nn
from p1model.calibration import LowRankSemanticCalibration
from p1model.dense_residual import DenseResidualCalibration

ARMS=('adaptive_mix','constant_mix')

class FrozenEndpointMixture(nn.Module):
    def __init__(self,dim,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('unknown mixture arm')
        self.embedding_dim=dim;self.arm=arm
        self.base=LowRankSemanticCalibration(dim,32)
        self.expert=DenseResidualCalibration(dim,'dense_info')
        self.base.eval();self.expert.eval() # Preserve canonical kernel path; immutable via no_grad + exclusion from optimizer.
        self.network=nn.Sequential(nn.Linear(4,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def train(self,mode=True):
        super().train(mode);self.base.eval();self.expert.eval();return self

    def endpoints(self,features,texts):
        if features.ndim!=4 or features.shape[1]!=2*self.embedding_dim:raise ValueError('packed native+dense required')
        with torch.no_grad():
            b=self.base(features[:,:self.embedding_dim].contiguous(memory_format=torch.channels_last),texts)
            e=self.expert(features,texts)
        return b,e

    @staticmethod
    def signals(b,e):
        # Common class shift invariant; class-agnostic, no labels or water/crop IDs.
        bv=b.topk(2,dim=1).values;ev=e.topk(2,dim=1).values
        p=b.softmax(1);q=e.softmax(1);middle=(p+q)*.5
        js=.5*((p*(p.clamp_min(1e-30).log()-middle.clamp_min(1e-30).log())).sum(1)+
                (q*(q.clamp_min(1e-30).log()-middle.clamp_min(1e-30).log())).sum(1))
        agree=(b.argmax(1)==e.argmax(1)).to(b.dtype)
        return torch.stack(((bv[:,0]-bv[:,1])*.07,(ev[:,0]-ev[:,1])*.07,js,agree),dim=-1)

    def forward(self,features,texts,gate_override=None):
        b,e=self.endpoints(features,texts)
        if gate_override is not None:
            if gate_override==0:return b
            if gate_override==1:return e
            raise ValueError('only endpoint witnesses 0/1 allowed')
        inputs=self.signals(b,e)
        if self.arm=='constant_mix':inputs=torch.zeros_like(inputs)
        gate=self.network(inputs).sigmoid().squeeze(-1).unsqueeze(1)
        return b+gate*(e-b)

    @property
    def trainable_parameter_count(self):return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
