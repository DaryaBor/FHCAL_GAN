#!/usr/bin/env python3
"""Conditional Flow Matching in original FHCal_GMM parameter space (27 + log total)."""
import argparse
import csv
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from .core import MODULES,load_three, add_data_args, data_source
from .parameter_flow import ParameterVelocity,loss
from .gmm_adapter import CODE_DIM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    p.add_argument('--fits',type=Path,default=Path('experiments/gmm_fits/fitbank.npz'))
    p.add_argument('--out',type=Path,default=Path('experiments/gmm_flow'))
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--width',type=int,default=256)
    p.add_argument('--blocks',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-other-kinematics',action='store_true')
    p.add_argument('--limit-batches',type=int,default=0,help='Smoke tests only')
    a=p.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else a.device if a.device!='auto' else 'cpu')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    with np.load(a.fits,allow_pickle=False) as bank:
        fit_meta=json.loads(str(bank['metadata']))
        ti=bank['train_indices'].copy();vi=bank['val_indices'].copy()
        target_train=bank['codes_train'].copy();target_val=bank['codes_val'].copy()
    if fit_meta['data_seed']!=a.seed:raise ValueError('Fit split seed and training seed differ')
    data=load_three(data_source(a),seed=a.seed,max_per_module=fit_meta['max_per_module'],
              allow_other_kinematics=a.allow_other_kinematics)
    if data.metadata!=fit_meta['source_metadata']:raise ValueError('Dataset files differ from original GMM fits')
    if not set(ti).issubset(set(data.train_idx)) or not np.array_equal(vi,data.val_idx):
        raise ValueError('Fit bank split train/val differs from expected held-out split')
    if target_train.shape!=(len(ti),CODE_DIM) or target_val.shape!=(len(vi),CODE_DIM):raise ValueError('Bad GMM codes')
    if not np.isfinite(target_train).all() or not np.isfinite(target_val).all():raise ValueError('Invalid GMM codes')
    train_mean=target_train.mean(axis=0).astype(np.float32)
    train_std=np.maximum(target_train.std(axis=0),0.05).astype(np.float32)
    ctrain=torch.tensor((target_train-train_mean)/train_std,dtype=torch.float32)
    cval=torch.tensor((target_val-train_mean)/train_std,dtype=torch.float32)
    train=DataLoader(TensorDataset(ctrain,data.conditions[ti]),batch_size=a.batch_size,shuffle=True)
    val=DataLoader(TensorDataset(cval,data.conditions[vi]),batch_size=a.batch_size,shuffle=False)
    model=ParameterVelocity(width=a.width,blocks=a.blocks).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    a.out.mkdir(parents=True,exist_ok=True)
    last_path=a.out/'last.pt';history=[];best=float('inf');best_epoch=0;start=1
    fit_digest=__import__('hashlib').sha256(a.fits.read_bytes()).hexdigest()
    if a.resume and last_path.is_file():
        ckpt=torch.load(last_path,map_location='cpu',weights_only=True)
        if ckpt['fits_sha256']!=fit_digest or ckpt['model_config']!={'width':a.width,'blocks':a.blocks}:
            raise ValueError('Fit bank/model differs from checkpoint')
        model.load_state_dict(ckpt['model_state']);opt.load_state_dict(ckpt['optimizer_state']);
        history=ckpt['history'];best=ckpt['best_val'];best_epoch=ckpt['best_epoch'];start=ckpt['epoch']+1
        torch.set_rng_state(ckpt['rng_cpu'])
        if device.type=='cuda' and ckpt.get('rng_cuda') is not None:torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
    print('DEVICE',device,'TRAIN',len(ti),'VAL',len(vi),'TEST HELD OUT',len(data.test_idx),
          'MODEL PARAMS',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(start,a.epochs+1):
        model.train();tr=0.;tn=0
        for i,(target,cond) in enumerate(train):
            if a.limit_batches and i>=a.limit_batches:break
            target=target.to(device);cond=cond.to(device)
            opt.zero_grad(set_to_none=True)
            l=loss(model,target,cond);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step();tr+=l.item()*len(target);tn+=len(target)
        model.eval();vv=0.;vn=0
        rng=torch.Generator(device=device.type).manual_seed(a.seed+100_000)
        with torch.no_grad():
            for i,(target,cond) in enumerate(val):
                if a.limit_batches and i>=a.limit_batches:break
                target=target.to(device);cond=cond.to(device)
                l=loss(model,target,cond,rng);vv+=l.item()*len(target);vn+=len(target)
        tr/=tn;vv/=vn
        history.append({'epoch':epoch,'train_loss':tr,'val_loss':vv})
        print(f'EPOCH {epoch:03d} train={tr:.6f} validation={vv:.6f}',flush=True)
        shared={'model_state':model.state_dict(),'model_config':{'width':a.width,'blocks':a.blocks},
                'normalization':{'scale':data.scale,'point_mean':data.point_mean.tolist(),
                                 'point_std':data.point_std.tolist()},
                'code_mean':train_mean.tolist(),'code_std':train_std.tolist(),
                'data_seed':a.seed,'max_per_module':fit_meta['max_per_module'],
                'allow_other_kinematics':a.allow_other_kinematics,'metadata':data.metadata,
                'fit_metadata':fit_meta,'fits_sha256':fit_digest,'epoch':epoch,
                'representation':'fhcal-gmm-k4-integrated-central-plus-log-total'}
        def save(obj,path):
            tmp=path.with_suffix('.pt.tmp');torch.save(obj,tmp);os.replace(tmp,path)
        if vv<best:
            best=vv;best_epoch=epoch
            save({**shared,'best_val':best,'best_epoch':best_epoch},a.out/'best.pt')
        save({**shared,'best_val':best,'best_epoch':best_epoch,'optimizer_state':opt.state_dict(),
              'history':history,'rng_cpu':torch.get_rng_state(),
              'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None},last_path)
        with (a.out/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,['epoch','train_loss','val_loss']);writer.writeheader();writer.writerows(history)
        (a.out/'config.json').write_text(json.dumps({**vars(a),'data_source':str(data_source(a)),
                      'fits':str(a.fits),'out':str(a.out),'training_codes':len(ti),
                      'validation_codes':len(vi)},indent=2,default=str))
    print('Best epoch',best_epoch,'best validation',best)

if __name__=='__main__':main()
