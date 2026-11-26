#!/usr/bin/env python3
"""Fit ORIGINAL FHCal_GMM K=4 parameters on TRAIN and VALIDATION ONLY.

Test events are never fitted/used. Output bank is private derived data; not Git-tracked.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
import torch

from .core import MODULES,load_three,decode, add_data_args, data_source
from .gmm_adapter import PARAM_DIM,CODE_DIM,K,MIN_SCALES,author_modules,flat_from_grid,fit_one,reference_path

_WORKER={}

def _initialize():
    p,modules,fit_fn,_,sha=author_modules()
    _WORKER.update(projector=p,fit_fn=fit_fn)

def _work(job):
    index,response,seed,restarts,maxiter=job
    code,info=fit_one(_WORKER['projector'],_WORKER['fit_fn'],response,seed+int(index)*7919,restarts,maxiter)
    return code,info

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument('--out',type=Path,default=Path('experiments/gmm_fits/fitbank.npz'))
    parser.add_argument('--max-per-module',type=int,default=600,help='Same selected N per module as direct model; 0=full dataset')
    parser.add_argument('--train-per-module',type=int,default=0,help='0=all selected train events. Pilot restriction is possible but breaks matched-size comparison.')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--restarts',type=int,default=1)
    parser.add_argument('--maxiter',type=int,default=100)
    parser.add_argument('--allow-other-kinematics',action='store_true')
    a=parser.parse_args()
    if a.out.exists():raise FileExistsError(f'{a.out} exists; preserve fitted codes. Choose another --out to refit.')
    if min(a.workers,a.restarts,a.maxiter)<1:raise ValueError('Workers/restarts/maxiter must be positive')
    data=load_three(data_source(a),seed=a.seed,max_per_module=a.max_per_module,
                    allow_other_kinematics=a.allow_other_kinematics)
    projector,modules,fit_fn,_,geometry_sha=author_modules()
    full_indices_train=data.train_idx
    if a.train_per_module:
        labels=data.module_ids.numpy()
        rng=np.random.default_rng(a.seed+817)
        selection=[]
        for mi in range(3):
            available=full_indices_train[labels[full_indices_train]==mi]
            selection.extend(rng.choice(available,size=min(a.train_per_module,len(available)),replace=False).tolist())
        train_indices=np.array(sorted(selection),dtype=np.int64)
    else:train_indices=full_indices_train.astype(np.int64)
    val_indices=data.val_idx.astype(np.int64)
    print('FITTING train',len(train_indices),'val',len(val_indices),'K',K,'params',PARAM_DIM,flush=True)
    indices=np.r_[train_indices,val_indices]
    energy=decode(data.x[indices],data.scale).numpy()
    shape=flat_from_grid(energy,modules)
    raw_digest=sha256(np.ascontiguousarray(shape).tobytes()).hexdigest()
    jobs=((int(i),shape[k],a.seed,a.restarts,a.maxiter) for k,i in enumerate(indices))
    timestamp=time.perf_counter()
    codes=[];status=[];losses=[];converged=[]
    if a.workers==1:
        _initialize()
        iterator=map(_work,jobs)
        for k,(code,details) in enumerate(iterator):
            codes.append(code);status.append(details['status']);losses.append(details['fit_loss']);converged.append(details.get('converged',False))
            if (k+1)%100==0:print('FIT',k+1,'/',len(indices),'seconds',round(time.perf_counter()-timestamp,1),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers,initializer=_initialize) as pool:
            for k,(code,details) in enumerate(pool.map(_work,jobs,chunksize=4)):
                codes.append(code);status.append(details['status']);losses.append(details['fit_loss']);converged.append(details.get('converged',False))
                if (k+1)%100==0:print('FIT',k+1,'/',len(indices),'seconds',round(time.perf_counter()-timestamp,1),flush=True)
    codes=np.stack(codes).astype(np.float32)
    train_count=len(train_indices)
    metadata={'data_seed':a.seed,'max_per_module':a.max_per_module,'train_per_module':a.train_per_module,
              'source_metadata':data.metadata,'n_fitted_train':len(train_indices),'n_fitted_val':len(val_indices),'restarts':a.restarts,'maxiter':a.maxiter,
              'min_scales':MIN_SCALES,'components':K,'param_dim':PARAM_DIM,'code_dim':CODE_DIM,
              'geometry_sha256':geometry_sha,'response_sha256':raw_digest,
              'original_repository':'https://github.com/voboch/FHCal_GMM',
              'reference_path_used':str(reference_path()),
              'fit_seconds':time.perf_counter()-timestamp,
              'note':'original FHCal_GMM SectionProjector and fit_mixture on central 34x7 physical sections, no test examples'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    # No test codes included, so held-out events stay strictly unseen by the GMM fitter.
    np.savez_compressed(a.out,train_indices=train_indices,val_indices=val_indices,
        codes_train=codes[:train_count],codes_val=codes[train_count:],
        fit_status=np.array(status),fit_loss=np.array(losses),fit_converged=np.array(converged),
        metadata=np.array(json.dumps(metadata,sort_keys=True)))
    print('SAVED',a.out,'fit seconds',round(metadata['fit_seconds'],1),flush=True)

if __name__=='__main__':main()
