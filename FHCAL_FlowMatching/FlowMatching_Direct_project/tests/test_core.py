import torch
import numpy as np
from flow_pipeline.core import VelocityNet, SHAPE, to_central, fm_loss, geometry_mask, integrate

def test_geometry():
    raw=np.zeros((2,7,9,10),np.float32)
    raw[:,:,2,:7]=2
    # ensure hole is empty
    raw[:,3,4,:]=0
    assert to_central(raw).shape==(2,*SHAPE)
    assert geometry_mask().sum().item()==34*7

def test_forward_and_sampling():
    model=VelocityNet(width=16,blocks=1)
    x=torch.randn(2,*SHAPE) * geometry_mask()
    c=torch.zeros(2,5)
    t=torch.rand(2)
    assert model(x,t,c).shape==x.shape
    loss=fm_loss(model,x,c);loss.backward()
    with torch.no_grad():
        generated=integrate(model,torch.randn_like(x)*geometry_mask(),c,steps=2)
    assert generated.shape==x.shape
