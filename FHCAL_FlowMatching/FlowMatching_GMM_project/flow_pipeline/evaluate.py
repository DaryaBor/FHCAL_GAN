#!/usr/bin/env python3
"""Inference of parameter CFM via the ORIGINAL voboch/FHCal_GMM projector + decoder."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset,TensorDataset
from .core import MODULES,load_three,decode
from .gmm_adapter import author_modules,decode_codes,CODE_DIM
from .parameter_flow import ParameterVelocity,integrate
from .metrics import save_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for module in MODULES:p.add_argument(f'--data{module}',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('experiments/gmm_flow/best.pt'))
    p.add_argument('--out',type=Path,default=Path('results/gmm'))
    p.add_argument('--num-per-module',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--solver',choices=['euler','heun'],default='euler')
    p.add_argument('--packets',type=int,default=128,help='128 original packet decoder, 0=smooth integrated map')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else a.device if a.device!='auto' else 'cpu')
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    data=load_three({m:getattr(a,f'data{m}') for m in MODULES},
                    seed=ckpt['data_seed'],max_per_module=ckpt['max_per_module'],
                    stats=ckpt['normalization'],allow_other_kinematics=ckpt['allow_other_kinematics'])
    if data.metadata!=ckpt['metadata']:raise ValueError('Dataset mismatch with saved model')
    projector,modules,_,packet_fn,geometry_sha=author_modules()
    if geometry_sha != ckpt['fit_metadata']['geometry_sha256']:
        raise ValueError('Detector geometry in FHCal_GMM reference changed since fitting')
    model=ParameterVelocity(**ckpt['model_config']).to(dev)
    model.load_state_dict(ckpt['model_state']);model.eval()
    mean=np.asarray(ckpt['code_mean'],dtype=np.float64)
    std=np.asarray(ckpt['code_std'],dtype=np.float64)
    true_all=[];fake_all=[];labels=[];seconds=0.0
    for mi,module in enumerate(MODULES):
        indices=data.test_idx[data.module_ids.numpy()[data.test_idx]==mi][:a.num_per_module]
        if len(indices)==0:raise ValueError(f'No test events for module {module}')
        ds=Subset(TensorDataset(data.x,data.conditions),indices.tolist())
        for b,(x,cond) in enumerate(DataLoader(ds,batch_size=a.batch_size)):
            cond=cond.to(dev)
            if dev.type=='cuda':torch.cuda.synchronize()
            t0=time.perf_counter()
            with torch.no_grad():
                initial=torch.randn((len(x),CODE_DIM),device=dev)
                generated=integrate(model,initial,cond,steps=a.steps,solver=a.solver)
                codes=generated.cpu().numpy().astype(np.float64)*std+mean
                fake=decode_codes(codes,projector,modules,packet_fn,packets=a.packets,
                                  seed=a.seed+mi*100_003+b)
            if dev.type=='cuda':torch.cuda.synchronize()
            seconds+=time.perf_counter()-t0
            fake_all.append(fake)
            true_all.append(decode(x,data.scale).numpy())
            labels.extend([module]*len(x))
    real=np.concatenate(true_all);fake=np.concatenate(fake_all);labels=np.asarray(labels)
    save_report(real,fake,labels,a.out,{'experiment':'original-fhcal-gmm-k4-parameter-flow',
                'decoder':'packet' if a.packets>0 else 'smooth','packets_per_component':a.packets,
                'geometry_sha256':geometry_sha,'n_train_fitted':ckpt['fit_metadata']['n_fitted_train'],
                'best_epoch':ckpt.get('best_epoch'),'checkpoint_epoch':ckpt['epoch'],
                'nfe':a.steps*(2 if a.solver=='heun' else 1),'solver':a.solver,'steps':a.steps,
                'device':str(dev),'num_events':len(fake),
                'generation_seconds_including_cpu_decode':seconds,'events_per_second':len(fake)/seconds,
                'data_metadata':data.metadata,'warning':'KS p-values on zero-inflated layers are descriptive; GMM smoother and packet postprocessing differ.'})
    print('SAVED',a.out,flush=True)

if __name__=='__main__':main()
