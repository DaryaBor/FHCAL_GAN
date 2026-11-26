import numpy as np
import torch
from flow_pipeline.parameter_flow import ParameterVelocity,DIM,loss,integrate
from flow_pipeline.gmm_adapter import author_modules,flat_from_grid,grid_from_flat,decode_codes,fit_one,CODE_DIM,canonicalize_params

def test_model():
    net=ParameterVelocity(width=32,blocks=1)
    x=torch.randn(4,DIM);c=torch.randn(4,5)
    assert net(x,torch.rand(4),c).shape==(4,DIM)
    loss(net,x,c).backward()
    assert integrate(net,torch.randn_like(x),c,steps=2).shape==x.shape

def test_original_projector_and_decoder():
    projector,modules,fitter,packet,sha=author_modules()
    assert len(modules)==34 and len(projector.lower)==238
    event=np.zeros((1,7,7,5),dtype=np.float32)
    event[:,3,2,2]=4
    event[:,3,3,3]=2
    flat=flat_from_grid(event,modules)
    assert flat.shape==(1,238)
    np.testing.assert_allclose(grid_from_flat(flat,modules),event)
    z,meta=fit_one(projector,fitter,flat[0],13,restarts=1,maxiter=4)
    assert z.shape==(CODE_DIM,) and np.isfinite(z).all()
    gen=decode_codes(z[None],projector,modules,packet,packets=16)
    assert gen.shape==event.shape
    np.testing.assert_allclose(gen.sum(),event.sum(),rtol=1e-5,atol=1e-5)
    assert (gen>=0).all()
    assert np.all(gen[:, :, 3, 2] == 0)


def test_canonicalize_permutation_invariance():
    projector,mods,_,_,_=author_modules()
    rng=np.random.default_rng(42)
    raw=np.r_[rng.uniform(-20,20,size=12),rng.uniform(-.3,1,size=12),rng.normal(size=3)]
    original=projector.mixture(raw,4)[0]
    canonical=projector.mixture(canonicalize_params(raw),4)[0]
    np.testing.assert_allclose(original,canonical,rtol=1e-5,atol=1e-7)
