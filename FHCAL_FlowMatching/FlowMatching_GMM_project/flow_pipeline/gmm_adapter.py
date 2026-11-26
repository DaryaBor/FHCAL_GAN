"""Adapter for the ORIGINAL voboch/FHCal_GMM projector, fitter, packet decoder.

No local reimplementation of its mathematical method. The external source is
checked out by scripts/get_reference.sh at project root and imported from reference_FHCal_GMM/src.
This experiment limits the original readout to 34 central modules x 7 layers.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os
import sys
import numpy as np

K=4
PARAM_DIM=7*K-1
CODE_DIM=PARAM_DIM+1    # log1p(total central energy)
MIN_SCALES=(2.5,2.5,0.175)
EXPECTED_REFERENCE_COMMIT='84fd872c03de9ed4bd03010c894150e4d8042bbd'


def reference_path():
    p=Path(os.environ.get('FHCAL_GMM_REFERENCE',Path(__file__).resolve().parents[1]/'reference_FHCal_GMM'))
    if not (p/'src/pipeline/gaussian_projection.py').is_file():
        raise FileNotFoundError(f'FHCal_GMM reference not found at {p}. Run bash scripts/get_reference.sh')
    return p


def author_modules():
    source=reference_path()/'src'
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    from pipeline.gaussian_projection import SectionProjector,fit_mixture,section_bounds
    from pipeline.projection_geometry import load_projection_geometry
    from conditional_gmm_baseline import sample_packet_fractions
    geometry=load_projection_geometry()
    modules=tuple(m for m in geometry.modules if m.layers==7 and 2<=m.legacy_col<=6)
    if len(modules)!=34 or any(m.legacy_row==3 and m.legacy_col==4 for m in modules):
        raise ValueError('Expected 34 seven-layer modules and a hole at (row=3,col=4)')
    lower,upper=section_bounds(SimpleNamespace(modules=modules))
    projector=SectionProjector(lower,upper,'integrated')
    return projector,modules,fit_mixture,sample_packet_fractions,geometry.sha256


def flat_from_grid(events,modules):
    """[N, depth=7, row=7, central_col=5] -> [N, 238] in author module order."""
    if events.ndim!=4 or tuple(events.shape[1:])!=(7,7,5):
        raise ValueError(f'Expected (N,7,7,5), got {events.shape}')
    return np.stack([events[:,layer,m.legacy_row,m.legacy_col-2]
                     for m in modules for layer in range(m.layers)],axis=1).astype(np.float64)


def grid_from_flat(fractions,modules):
    if fractions.ndim!=2 or fractions.shape[1]!=238:raise ValueError(f'Expected (N,238), got {fractions.shape}')
    output=np.zeros((len(fractions),7,7,5),dtype=np.float32)
    k=0
    for m in modules:
        for layer in range(m.layers):
            output[:,layer,m.legacy_row,m.legacy_col-2]=fractions[:,k]
            k+=1
    return output


def canonicalize_params(params):
    """Deterministic component order. Same physical mixture; different parameter labels.

    This sorting is OUR adaptation for ML stability (not part of the upstream fitter).
    """
    params=np.asarray(params,dtype=np.float64)
    means=params[:3*K].reshape(K,3)
    log_scales=params[3*K:6*K].reshape(K,3)
    logits=np.r_[params[6*K:],0.0]
    # z primary, then x and y; deterministic for fixed fitted parameters.
    order=np.lexsort((means[:,1],means[:,0],means[:,2]))
    new_logits=logits[order][:-1]-logits[order][-1]
    return np.r_[means[order].ravel(),log_scales[order].ravel(),new_logits].astype(np.float32)


def fit_one(projector,fit_mixture,energy,seed,restarts=1,maxiter=100):
    """Original GMM L-BFGS-B fitting; return 27 params and log1p(total)."""
    energy=np.asarray(energy,dtype=np.float64)
    if len(energy)!=238 or np.any(energy<0) or not np.isfinite(energy).all():
        raise ValueError('Invalid event energy vector')
    total=float(energy.sum())
    if total==0:return np.zeros(CODE_DIM,dtype=np.float32),{'status':-1,'fit_loss':0.0}
    fitted=fit_mixture(projector,energy/total,K,restarts,maxiter,seed,MIN_SCALES,trace_restarts=True)
    chosen=[r for r in fitted.restart_trace if r['selected']]
    if len(chosen)!=1 or chosen[0]['parameters'] is None:
        raise RuntimeError('No valid selected parameter vector in original fitter')
    params=canonicalize_params(chosen[0]['parameters'])
    code=np.r_[params,np.float32(np.log1p(total))]
    return code,{'status':int(fitted.status),'fit_loss':float(fitted.loss),'converged':bool(fitted.converged)}


def decode_codes(codes,projector,modules,packet_fn,packets=128,seed=2026):
    """Original author's SectionProjector and packet sampler, restricted to central sections."""
    rng=np.random.default_rng(seed)
    low=np.asarray(projector.lower).min(axis=0)
    high=np.asarray(projector.upper).max(axis=0)
    min_scale=np.asarray(MIN_SCALES)
    max_scale=np.asarray(projector.extent)
    out=np.zeros((len(codes),238),dtype=np.float64)
    for i,code in enumerate(np.asarray(codes,dtype=np.float64)):
        params=code[:PARAM_DIM].copy()
        total=float(np.expm1(np.clip(code[-1],0,18)))
        if total<=0:continue
        params[:3*K]=np.clip(params[:3*K].reshape(K,3),low,high).ravel()
        params[3*K:6*K]=np.clip(params[3*K:6*K].reshape(K,3),np.log(min_scale),np.log(max_scale)).ravel()
        params[6*K:]=np.clip(params[6*K:],-24,24)
        smooth,components,_,weights=projector.mixture(params,K)
        if packets>0:
            fractions=packet_fn(components,weights,packets,rng)
        else:
            fractions=smooth
        out[i]=total*fractions
    return grid_from_flat(out,modules)
