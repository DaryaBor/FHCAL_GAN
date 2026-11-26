#!/usr/bin/env python3
"""Split one mixed NPZ of fixed muon beams into 8/16/19 files.

Uses beam DIRECTION (not magnitude) to label events. The known directions
are a hypothesis from the simulation configuration and checked with a strict
angular tolerance. Never silently assign ambiguous events.
"""
from pathlib import Path
import argparse
import json
import numpy as np

BEAMS = {8:(10.70,128.48), 16:(3.0,0.0), 19:(12.0,180.0)}

def unit_vector(theta,phi):
    th=np.deg2rad(theta);ph=np.deg2rad(phi)
    return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph),np.cos(th)])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='One NPZ containing all three beams')
    p.add_argument('--out',default='data')
    p.add_argument('--max-angle-deg',type=float,default=2.0)
    a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    with np.load(a.input,allow_pickle=False) as ds:
        needed=('EnergyDeposit','ParticlePDG','ParticleMomentum','ParticlePoint')
        if any(k not in ds.files for k in needed):
            raise ValueError(f'NPZ needs {needed}; keys are {ds.files}')
        mom=ds['ParticleMomentum'].astype(np.float64)
        pdg=ds['ParticlePDG']
        if not np.all(pdg==-13):
            raise ValueError('Dataset contains non-mu+ events; refuse mixed PDGs')
        norms=np.linalg.norm(mom,axis=1)
        if np.any(norms<1e-8): raise ValueError('Zero momentum present')
        ref=np.stack([unit_vector(*BEAMS[m]) for m in (8,16,19)])
        cos=np.clip((mom/norms[:,None])@ref.T,-1,1)
        angles=np.rad2deg(np.arccos(cos))
        label=angles.argmin(axis=1)
        chosen=angles[np.arange(len(label)),label]
        if np.max(chosen)>a.max_angle_deg:
            bad=np.flatnonzero(chosen>a.max_angle_deg)
            raise ValueError(f'{len(bad)} events do not match any configured beam (max angle {chosen.max():.2f} degrees). '
                             f'First bad event {bad[0]} momentum {mom[bad[0]].tolist()}. '
                             'Check beam angles; do NOT split by guess.')
        info={'source':str(Path(a.input).name),'tolerance_degrees':a.max_angle_deg,'datasets':[]}
        for k,m in enumerate((8,16,19)):
            idx=np.flatnonzero(label==k)
            if not len(idx):raise ValueError(f'No events classified as module {m}')
            print(f'Module {m}: n={len(idx)}, mean p={norms[idx].mean():.4f}, max angle error={chosen[idx].max():.4f} deg',flush=True)
            path=root/f'fhcal_data_muons_{m}.npz'
            # Keep source spatial layout. Model training will crop to 7x5x7.
            np.savez_compressed(path, EnergyDeposit=ds['EnergyDeposit'][idx],
                                ParticlePoint=ds['ParticlePoint'][idx],
                                ParticleMomentum=ds['ParticleMomentum'][idx],
                                ParticlePDG=pdg[idx])
            info['datasets'].append({'module':m,'events':len(idx),'file':path.name,'p_mean':float(norms[idx].mean())})
    (root/'split_provenance.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Done. Inspect provenance in',root/'split_provenance.json')

if __name__=='__main__':main()
