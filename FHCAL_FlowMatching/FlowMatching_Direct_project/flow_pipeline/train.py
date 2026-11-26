#!/usr/bin/env python3
"""Direct conditional Flow Matching, central 34 modules x 7 layers, conditions 8/16/19."""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .core import MODULES, VelocityNet, fm_loss, load_three


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for module in MODULES:
        p.add_argument(f'--data{module}', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('experiments/direct'))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--max-per-module', type=int, default=0, help='0=all events, otherwise first randomly selected N events per module')
    p.add_argument('--limit-batches', type=int, default=0, help='0=all batches; smoke tests only')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-other-kinematics', action='store_true')
    return p.parse_args()


def atomic_save(obj,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)


def main():
    a=arguments()
    if a.epochs<1 or a.batch_size<1: raise ValueError('Bad epochs or batch size')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else a.device if a.device!='auto' else 'cpu')
    print('DEVICE',device,flush=True)
    a.out.mkdir(parents=True,exist_ok=True)
    last_path=a.out/'last.pt'
    prior=torch.load(last_path,weights_only=True,map_location='cpu') if a.resume and last_path.is_file() else None
    if prior and (prior['model_config']!={'width':a.width,'blocks':a.blocks} or prior['data_seed']!=a.seed or prior['max_per_module']!=a.max_per_module):
        raise ValueError('Resume configuration differs from saved training run')
    data=load_three({m:getattr(a,f'data{m}') for m in MODULES}, seed=a.seed,max_per_module=a.max_per_module,
                    stats=(prior['normalization'] if prior else None),allow_other_kinematics=a.allow_other_kinematics)
    if prior and data.metadata != prior['metadata']:
        raise ValueError('Dataset files or kinematics changed relative to checkpoint')
    stats={'scale':data.scale,'point_mean':data.point_mean.tolist(),'point_std':data.point_std.tolist()}
    model=VelocityNet(width=a.width,blocks=a.blocks).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    history=[];best=float('inf');best_epoch=0;start=1
    if prior:
        model.load_state_dict(prior['model_state']);optimizer.load_state_dict(prior['optimizer_state'])
        history=prior['history'];best=prior['best_val'];best_epoch=prior['best_epoch'];start=prior['epoch']+1
        torch.set_rng_state(prior['rng_cpu'])
        if device.type=='cuda' and prior['rng_cuda'] is not None:torch.cuda.set_rng_state_all(prior['rng_cuda'])
    labels=data.module_ids.numpy()
    count=np.bincount(labels[data.train_idx],minlength=3)
    sampler=WeightedRandomSampler(1.0/count[labels[data.train_idx]],num_samples=len(data.train_idx),replacement=True)
    train=DataLoader(data.subset('train'),batch_size=a.batch_size,sampler=sampler)
    val=DataLoader(data.subset('val'),batch_size=a.batch_size,shuffle=False)
    print('DATA train/val/test',len(data.train_idx),len(data.val_idx),len(data.test_idx),
          'EVENTS',[(z['module'],z['selected_events'],z['actual_p']) for z in data.metadata],
          'PARAMS',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(start,a.epochs+1):
        model.train();tr_total=0.;tr_n=0
        for i,(x,cond,_) in enumerate(train):
            if a.limit_batches and i>=a.limit_batches:break
            x=x.to(device);cond=cond.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=fm_loss(model,x,cond);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            tr_total+=loss.item()*len(x);tr_n+=len(x)
        model.eval();v_total=0.;v_n=0
        rng=torch.Generator(device=device.type).manual_seed(a.seed+100_000)
        with torch.no_grad():
            for i,(x,cond,_) in enumerate(val):
                if a.limit_batches and i>=a.limit_batches:break
                x=x.to(device);cond=cond.to(device)
                loss=fm_loss(model,x,cond,rng)
                v_total+=loss.item()*len(x);v_n+=len(x)
        tr=tr_total/tr_n;vv=v_total/v_n
        history.append({'epoch':epoch,'train_loss':tr,'val_loss':vv})
        print(f'EPOCH {epoch:03d} train={tr:.6f} validation={vv:.6f}',flush=True)
        shared={'model_state':model.state_dict(),'model_config':{'width':a.width,'blocks':a.blocks},
                'normalization':stats,'data_seed':a.seed,'max_per_module':a.max_per_module,
                'allow_other_kinematics':a.allow_other_kinematics,'metadata':data.metadata,
                'epoch':epoch,'representation':'direct-sections'}
        if vv<best:
            best=vv;best_epoch=epoch
            atomic_save({**shared,'best_val':best,'best_epoch':best_epoch},a.out/'best.pt')
        atomic_save({**shared,'best_val':best,'best_epoch':best_epoch,
                     'optimizer_state':optimizer.state_dict(),'history':history,
                     'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None},last_path)
        with (a.out/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','val_loss']);writer.writeheader();writer.writerows(history)
        (a.out/'config.json').write_text(json.dumps({**vars(a),'out':str(a.out),
                      **{f'data{m}':str(getattr(a,f'data{m}')) for m in MODULES},'data_metadata':data.metadata},indent=2,ensure_ascii=False))
    print('Best epoch',best_epoch,'best validation',best)

if __name__=='__main__':main()
