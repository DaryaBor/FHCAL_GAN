"""Flow Matching over 27 GMM parameters plus log1p(total) = 28 coordinates."""
import math
import torch
from torch import nn
import torch.nn.functional as F

DIM=28

class TimeEncoding(nn.Module):
    def __init__(self,width=128):
        super().__init__();self.width=width
    def forward(self,t):
        n=self.width//2
        factor=torch.exp(torch.arange(n,device=t.device,dtype=t.dtype)*(-math.log(10000)/max(1,n-1)))
        a=t[:,None]*factor[None]
        return torch.cat([torch.sin(a),torch.cos(a)],dim=-1)

class ResidualMLP(nn.Module):
    def __init__(self,width=256):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.fc1=nn.Linear(width,width)
        self.fc2=nn.Linear(width,width)
        nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
    def forward(self,x,condition):
        return (x+self.fc2(F.silu(self.fc1(F.silu(self.norm(x))+condition))))/math.sqrt(2)

class ParameterVelocity(nn.Module):
    def __init__(self,width=256,blocks=4,time_dim=128):
        super().__init__()
        self.time=nn.Sequential(TimeEncoding(time_dim),nn.Linear(time_dim,time_dim),nn.SiLU())
        self.cond=nn.Sequential(nn.Linear(5,time_dim),nn.SiLU(),nn.Linear(time_dim,time_dim))
        self.inp=nn.Linear(DIM,width)
        self.condition=nn.Linear(time_dim,width)
        self.blocks=nn.ModuleList(ResidualMLP(width) for _ in range(blocks))
        self.out=nn.Sequential(nn.LayerNorm(width),nn.SiLU(),nn.Linear(width,DIM))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
    def forward(self,x,t,cond):
        if x.ndim!=2 or x.shape[1]!=DIM or cond.shape!=(len(x),5):raise ValueError('Expected [B,28] and [B,5]')
        k=self.condition(self.time(t)+self.cond(cond))
        h=self.inp(x)
        for block in self.blocks:h=block(h,k)
        return self.out(h)


def loss(model,target,cond,generator=None):
    prior=torch.randn(target.shape,dtype=target.dtype,device=target.device,generator=generator)
    t=torch.rand((len(target),),device=target.device,generator=generator)
    xt=prior*(1-t[:,None])+target*t[:,None]
    return F.mse_loss(model(xt,t,cond),target-prior)

@torch.no_grad()
def integrate(model,x0,cond,steps=20,solver='euler'):
    if solver not in ('euler','heun') or steps<1:raise ValueError('Invalid ODE setup')
    x=x0;h=1.0/steps
    for j in range(steps):
        v=model(x,x.new_full((len(x),),j/steps),cond)
        if solver=='euler':x=x+h*v
        else:
            v_next=model(x+h*v,x.new_full((len(x),),(j+1)/steps),cond)
            x=x+(h/2)*(v+v_next)
    return x
