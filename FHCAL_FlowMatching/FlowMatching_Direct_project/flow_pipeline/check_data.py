#!/usr/bin/env python3
"""Validate the three physical muon scenarios and central 7x5x7 crop."""
import argparse
from pathlib import Path
from .core import MODULES,load_three, add_data_args, data_source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-per-module',type=int,default=600)
    p.add_argument('--allow-other-kinematics',action='store_true')
    a=p.parse_args()
    d=load_three(data_source(a),
                 seed=a.seed,max_per_module=a.max_per_module,
                 allow_other_kinematics=a.allow_other_kinematics)
    print('Shape:',tuple(d.x.shape),'Conditions:',tuple(d.conditions.shape))
    print('Split:',len(d.train_idx),len(d.val_idx),len(d.test_idx))
    print('Metadata:')
    for m in d.metadata:
        print(m)

if __name__=='__main__':main()
