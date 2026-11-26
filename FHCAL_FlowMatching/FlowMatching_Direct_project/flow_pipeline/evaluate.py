#!/usr/bin/env python3
"""Generate and compare central FHCal responses using a saved direct CNN model."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from .core import MODULES, VelocityNet, decode, integrate, load_three, geometry_mask
from .metrics import save_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for m in MODULES:p.add_argument(f'--data{m}',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('experiments/direct/best.pt'))
    p.add_argument('--out',type=Path,default=Path('results/direct'))
    p.add_argument('--num-per-module',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--solver',choices=['euler','heun'],default='euler')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else a.device if a.device!='auto' else 'cpu')
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    data=load_three({m:getattr(a,f'data{m}') for m in MODULES},seed=ckpt['data_seed'],max_per_module=ckpt['max_per_module'],
           stats=ckpt['normalization'],allow_other_kinematics=ckpt['allow_other_kinematics'])
    if data.metadata!=ckpt['metadata']:raise ValueError('Input files differ from training data')
    model=VelocityNet(**ckpt['model_config']).to(dev)
    model.load_state_dict(ckpt['model_state']);model.eval()
    true_all=[];fake_all=[];labels=[];seconds=0.0
    for mi,m in enumerate(MODULES):
        indices=data.test_idx[data.module_ids.numpy()[data.test_idx]==mi][:a.num_per_module]
        if len(indices)==0:raise ValueError(f'No test events for module {m}')
        loader=DataLoader(Subset(TensorDataset(data.x,data.conditions),indices.tolist()),batch_size=a.batch_size)
        for x,cond in loader:
            cond=cond.to(dev)
            if dev.type=='cuda':torch.cuda.synchronize()
            tick=time.perf_counter()
            with torch.no_grad():
                noise=torch.randn((len(x),7,7,5),device=dev)*geometry_mask(dev)
                latent=integrate(model,noise,cond,steps=a.steps,solver=a.solver)
                fake=decode(latent,data.scale).cpu().numpy()
            if dev.type=='cuda':torch.cuda.synchronize()
            seconds+=time.perf_counter()-tick
            fake_all.append(fake)
            true_all.append(decode(x,data.scale).numpy())
            labels.extend([m]*len(x))
    fake=np.concatenate(fake_all);real=np.concatenate(true_all);labels=np.array(labels)
    save_report(real,fake,labels,a.out,{'experiment':'direct-cnn','best_epoch':ckpt.get('best_epoch'),
                  'checkpoint_epoch':ckpt['epoch'],'nfe':a.steps*(2 if a.solver=='heun' else 1),
                  'solver':a.solver,'steps':a.steps,'device':str(dev),'num_events':len(fake),
                  'generation_seconds':seconds,'events_per_second':len(fake)/seconds,
                  'data_metadata':data.metadata,'warning':'KS p-values on zero-inflated layers are descriptive.'})
    print('SAVED',a.out,flush=True)

if __name__=='__main__':main()
