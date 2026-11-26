"""Shared central FHCal preprocessing, muon conditions and direct CNN backbone."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

MODULES = (8, 16, 19)
# output tensor (depth, y, x), only central 5x7 modules and 7 sections
SHAPE = (7, 7, 5)
HOLE_Y, HOLE_X = 3, 2
# Nominal example kinematics only: the loader logs actual recorded momenta.
KINEMATICS = {8:(2.0,10.70,128.48), 16:(5.0,3.0,0.0), 19:(5.0,12.0,180.0)}


def geometry_mask(device=None, dtype=torch.float32):
    m = torch.ones((1, *SHAPE), dtype=dtype, device=device)
    m[:, :, HOLE_Y, HOLE_X] = 0.0
    return m


def to_central(raw):
    """Stored NPZ (N, y, x, depth) -> CNN tensor (N, depth, y, x)."""
    if raw.ndim != 4:
        raise ValueError(f'EnergyDeposit must have 4 dimensions: {raw.shape}')
    if tuple(raw.shape[1:]) == (7, 9, 10):
        selected = raw[:, :, 2:7, :7]  # 34 valid central modules + central hole
    elif tuple(raw.shape[1:]) == (7, 5, 7):
        selected = raw
    else:
        raise ValueError(f'Unsupported energy layout: {raw.shape}; expected (N,7,9,10) or (N,7,5,7)')
    result = np.ascontiguousarray(selected.transpose(0, 3, 1, 2), dtype=np.float32)
    if np.any(result < 0) or not np.isfinite(result).all():
        raise ValueError('Negative or nonfinite energy in dataset')
    if not np.all(result[:, :, HOLE_Y, HOLE_X] == 0):
        raise ValueError('The central hole is not empty: check mapping/geometrical conventions')
    return result


@dataclass
class ThreeData:
    x: torch.Tensor
    conditions: torch.Tensor
    module_ids: torch.Tensor
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    scale: float
    point_mean: np.ndarray
    point_std: np.ndarray
    metadata: list

    def subset(self, split):
        idx = getattr(self, f'{split}_idx')
        return torch.utils.data.Subset(torch.utils.data.TensorDataset(self.x, self.conditions, self.module_ids), idx.tolist())


def add_data_args(parser):
    """One combined NPZ (--data) or legacy three files (--data8/16/19)."""
    import argparse
    parser.add_argument('--data', type=Path, help='One combined muon NPZ with all 3 beam directions')
    for module in MODULES:
        parser.add_argument(f'--data{module}', type=Path, help=f'Legacy separate file for module {module}')


def data_source(args):
    """Validate CLI arguments; return combined Path or module->Path mapping."""
    separate = {m: getattr(args, f'data{m}') for m in MODULES}
    if args.data is not None:
        if any(v is not None for v in separate.values()):
            raise ValueError('Use --data OR --data8/16/19, not both')
        return Path(args.data)
    if not all(v is not None for v in separate.values()):
        raise ValueError('Provide --data combined.npz OR all of --data8, --data16, --data19')
    return separate


def beam_labels(momentum, max_angle_deg=2.0):
    """Classify muon beam by recorded momentum DIRECTION, not event order.

    Calibrated against the beam directions documented in this project. A mismatched
    beam is a fatal error, never silently allocated to the closest module.
    """
    mom = np.asarray(momentum, dtype=np.float64)
    if mom.ndim != 2 or mom.shape[1] != 3 or not np.isfinite(mom).all():
        raise ValueError('ParticleMomentum must be a finite (N,3) array')
    norms = np.linalg.norm(mom, axis=1)
    if not len(mom) or np.any(norms < 1e-8):
        raise ValueError('Empty or zero ParticleMomentum vectors')
    refs=[]
    for module in MODULES:
        _, theta, phi = KINEMATICS[module]
        th, ph = np.deg2rad([theta, phi])
        refs.append([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    ref=np.asarray(refs)
    angles=np.rad2deg(np.arccos(np.clip((mom/norms[:,None]) @ ref.T,-1,1)))
    closest=angles.argmin(axis=1)
    errors=angles[np.arange(len(closest)),closest]
    if np.any(errors>max_angle_deg):
        i=int(np.flatnonzero(errors>max_angle_deg)[0])
        raise ValueError(f'Cannot identify incident module for {np.count_nonzero(errors>max_angle_deg)} events; '
                         f'first index={i}, momentum={mom[i].tolist()}, nearest-angle={errors[i]:.3f} deg. '
                         'Verify simulation beam directions before training.')
    for i,mod in enumerate(MODULES):
        if not np.any(closest==i):
            raise ValueError(f'No events for module {mod} in combined file')
    return closest


def _finish_three(raw_parts,pos_parts,label_parts,train,val,test,metadata,stats):
    """Identical normalization and splitting for combined and three-file loading."""
    train_idx=np.concatenate(train); val_idx=np.concatenate(val);test_idx=np.concatenate(test)
    x_raw=np.concatenate(raw_parts, axis=0)
    points=np.concatenate(pos_parts, axis=0)
    labels=np.concatenate(label_parts, axis=0)
    if stats is None:
        active = x_raw[train_idx]
        positive=np.log1p(active[active>0])
        scale=float(max(np.quantile(positive,0.995),0.1)) if len(positive) else 0.1
        pmean=points[train_idx].mean(axis=0)
        pstd=points[train_idx].std(axis=0)
        pstd=np.maximum(pstd,1e-4)
    else:
        scale=float(stats['scale'])
        pmean=np.asarray(stats['point_mean'],dtype=np.float32)
        pstd=np.asarray(stats['point_std'],dtype=np.float32)
    x=np.log1p(x_raw)/scale
    cond=np.concatenate([np.eye(len(MODULES),dtype=np.float32)[labels],(points-pmean)/pstd],axis=1)
    assert tuple(x.shape[1:])==SHAPE and cond.shape[1]==5
    return ThreeData(torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)),
                     torch.from_numpy(np.ascontiguousarray(cond,dtype=np.float32)),
                     torch.from_numpy(labels.copy()),train_idx,val_idx,test_idx,
                     scale,pmean,pstd,metadata)


def load_combined(path, seed=42, max_per_module=0, stats=None, allow_other_kinematics=False):
    """One NPZ, one conditional model; stratify 8/16/19 by particle momentum.

    Only the requested pilot examples per beam are copied into model arrays.
    Reading compressed NPZ nevertheless requires decompressing EnergyDeposit once.
    """
    import hashlib
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Combined muon dataset does not exist: {p}')
    parts,points_parts,labels_parts,tr,va,te,metadata=[],[],[],[],[],[],[]
    cursor=0
    with np.load(p,allow_pickle=False) as ds:
        if not {'EnergyDeposit','ParticlePDG','ParticlePoint','ParticleMomentum'} <= set(ds.files):
            raise ValueError(f'Missing required keys from {p}; available: {ds.files}')
        pdg=ds['ParticlePDG']
        if not np.all(pdg==-13):
            raise ValueError(f'Expected only mu+ (PDG=-13), received {np.unique(pdg).tolist()}')
        pos=np.asarray(ds['ParticlePoint'],dtype=np.float32)
        mom=np.asarray(ds['ParticleMomentum'],dtype=np.float32)
        if pos.shape!=(len(pdg),3) or mom.shape!=(len(pdg),3):
            raise ValueError(f'Unexpected ParticlePoint/ParticleMomentum shapes: {pos.shape}, {mom.shape}')
        labels=beam_labels(mom)
        # Loading one zipped npz's EnergyDeposit can require gigabytes of RAM;
        # avoid allocating three additional complete copies before subsampling.
        raw=ds['EnergyDeposit']
        if len(raw)!=len(pdg):
            raise ValueError('EnergyDeposit and ParticlePDG event counts differ')
        with p.open('rb') as handle:
            digest=hashlib.file_digest(handle,'sha256').hexdigest()
        for mi,module in enumerate(MODULES):
            candidate=np.flatnonzero(labels==mi)
            rng=np.random.default_rng(seed+module*1009)
            chosen=rng.permutation(candidate)
            if max_per_module:
                chosen=chosen[:min(int(max_per_module),len(chosen))]
            n=len(chosen)
            if n<30:
                raise ValueError(f'Module {module} has too few events ({n})')
            beam=mom[candidate]
            actual_mom=beam.mean(axis=0)
            actual_p=float(np.linalg.norm(actual_mom))
            target_p,target_theta,target_phi=KINEMATICS[module]
            th,ph=np.deg2rad([target_theta,target_phi])
            ref=np.array([target_p*np.sin(th)*np.cos(ph),target_p*np.sin(th)*np.sin(ph),target_p*np.cos(th)])
            dot=np.clip(float(np.dot(actual_mom,ref)/(np.linalg.norm(actual_mom)*np.linalg.norm(ref))),-1.,1.)
            error=float(np.rad2deg(np.arccos(dot)))
            if error>2.0 and not allow_other_kinematics:
                raise ValueError(f'Module {module} direction mismatch ({error:.3f} degrees)')
            if abs(actual_p-target_p)>0.03:
                print(f'NOTE module {module}: actual p={actual_p:.4f} GeV/c, nominal p={target_p:.4f}; actual is used',flush=True)
            if np.max(np.abs(beam-beam[0]))>1e-3:
                raise ValueError(f'Multiple momenta inside module {module}; expected one fixed beam')
            event=to_central(raw[chosen])
            kept=float(np.sum(event,dtype=np.float64)/max(np.sum(raw[chosen],dtype=np.float64),1e-12))
            print(f'Module {module}: {n} of {len(candidate)} events, central energy retained={kept:.4f}, '
                  f'p={actual_p:.4f} GeV/c, angular diff={error:.4f} deg',flush=True)
            parts.append(event)
            points_parts.append(pos[chosen,:2].copy())
            labels_parts.append(np.full(n,mi,dtype=np.int64))
            ntr=int(n*.8); nva=int(n*.1)
            tr.append(np.arange(cursor,cursor+ntr))
            va.append(np.arange(cursor+ntr,cursor+ntr+nva))
            te.append(np.arange(cursor+ntr+nva,cursor+n))
            cursor+=n
            metadata.append({'file_sha256':digest,'module':module,'file':p.name,
                    'combined':True,'original_events':int(len(candidate)),'selected_events':n,
                    'central_energy_fraction':kept,
                    'momentum_xyz_example':mom[chosen[0]].tolist(),
                    'nominal_kinematics':{'p':target_p,'theta':target_theta,'phi':target_phi},
                    'actual_p':actual_p,'direction_error_deg':error})
    return _finish_three(parts,points_parts,labels_parts,tr,va,te,metadata,stats)


def load_three(paths, seed=42, max_per_module=0, stats=None, allow_other_kinematics=False):
    """Stratified 80/10/10 split per impact module. Fit normalization on train only.

    File 16/19 must be muons, never the older proton dataset.
    max_per_module=0 uses all data. stats from checkpoint prevents refitting.
    """
    if isinstance(paths, (str, Path)):
        return load_combined(paths, seed=seed, max_per_module=max_per_module,
                             stats=stats, allow_other_kinematics=allow_other_kinematics)
    if set(paths) != set(MODULES):
        raise ValueError(f'Provide one path for each module {MODULES}')
    raw_parts, pos_parts, label_parts, train, val, test, metadata = [], [], [], [], [], [], []
    cursor = 0
    for module in MODULES:
        p = Path(paths[module])
        if not p.is_file():
            raise FileNotFoundError(f'Missing muon dataset for module {module}: {p}')
        with np.load(p, allow_pickle=False) as data:
            if not {'EnergyDeposit','ParticlePDG','ParticlePoint','ParticleMomentum'} <= set(data.files):
                raise ValueError(f'Missing required keys in {p}')
            pdg = data['ParticlePDG']
            if not np.all(pdg == -13):
                raise ValueError(f'{p} is not pure mu+ (expected PDG -13); found {np.unique(pdg).tolist()}')
            raw = data['EnergyDeposit']
            pos = np.asarray(data['ParticlePoint'], dtype=np.float32)
            mom = np.asarray(data['ParticleMomentum'], dtype=np.float32)
            if pos.shape != (len(raw),3) or mom.shape != (len(raw),3):
                raise ValueError(f'Invalid particle feature shapes in {p}')
            target_p,target_theta,target_phi=KINEMATICS[module]
            theta=np.deg2rad(target_theta); phi=np.deg2rad(target_phi)
            ref=np.array([target_p*np.sin(theta)*np.cos(phi),
                          target_p*np.sin(theta)*np.sin(phi),target_p*np.cos(theta)])
            # A fixed momentum for each module is sufficient when the module ID is a condition.
            # The nominal |p| may differ between campaigns, so log it instead of silently
            # treating a 5 GeV dataset as if it were the nominal 2 GeV dataset.
            actual_mom=mom.mean(axis=0)
            actual_p=float(np.linalg.norm(actual_mom))
            dot=np.clip(float(np.dot(actual_mom,ref) / (np.linalg.norm(actual_mom)*np.linalg.norm(ref))),-1.,1.)
            direction_error=float(np.rad2deg(np.arccos(dot)))
            if direction_error>2.0 and not allow_other_kinematics:
                raise ValueError(f'{p} has beam direction {direction_error:.2f} degrees from the '
                                 f'expected direction for module {module}. Check mapping, or '
                                 'explicitly pass --allow-other-kinematics after inspection.')
            if abs(actual_p-target_p)>0.03:
                print(f'NOTE module {module}: actual p={actual_p:.4f} GeV/c, '
                      f'nominal p={target_p:.4f}. The actual recorded p is used; '
                      'report this as a distinct beam configuration.',flush=True)
            if np.max(np.abs(mom-mom[0]))>1e-3:
                raise ValueError(f'{p} contains multiple momenta within fixed module {module}')
            rng = np.random.default_rng(seed + module * 1009)
            ix = rng.permutation(len(raw))
            if max_per_module:
                ix = ix[:min(int(max_per_module),len(ix))]
            # Note: selected indices randomly ordered; split is immutable across priors.
            n = len(ix)
            if n < 30:
                raise ValueError(f'Module {module} has too few events ({n})')
            event = to_central(raw[ix])
            point = pos[ix,:2].copy()
            kept = float(np.sum(event, dtype=np.float64) / max(np.sum(raw[ix], dtype=np.float64),1e-12))
            print(f'Module {module}: {n} mu+ events; shape {event.shape}, retained central-energy fraction={kept:.4f}', flush=True)
            import hashlib
            with p.open('rb') as stream:
                file_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            metadata.append({'file_sha256':file_hash,'module':module,'file':p.name,'original_events':len(raw), 'selected_events':n,
                             'central_energy_fraction':kept,
                             'momentum_xyz_example':mom[ix[0]].tolist(),
                             'nominal_kinematics':{'p':target_p,'theta':target_theta,'phi':target_phi},
                             'actual_p':actual_p,'direction_error_deg':direction_error})
        n_train=int(n*.8); n_val=int(n*.1)
        raw_parts.append(event);pos_parts.append(point)
        label_parts.append(np.full(n,MODULES.index(module),dtype=np.int64))
        train.append(np.arange(cursor,cursor+n_train))
        val.append(np.arange(cursor+n_train,cursor+n_train+n_val))
        test.append(np.arange(cursor+n_train+n_val,cursor+n))
        cursor += n
    return _finish_three(raw_parts,pos_parts,label_parts,train,val,test,metadata,stats)


def decode(x,scale):
    e=torch.expm1(torch.clamp(x*scale,max=20.0)).clamp_min(0)
    return e * geometry_mask(x.device,x.dtype)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self,dim):
        super().__init__();self.dim=dim
    def forward(self,t):
        half=self.dim//2
        f=torch.exp(torch.arange(half,device=t.device,dtype=t.dtype)*(-math.log(10000)/max(half-1,1)))
        arg=t[:,None]*f[None]
        return torch.cat([torch.sin(arg),torch.cos(arg)],dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self,width,emb_dim):
        super().__init__()
        self.norm1=nn.GroupNorm(8,width)
        self.conv1=nn.Conv2d(width,width,3,padding=1)
        self.norm2=nn.GroupNorm(8,width)
        self.emb_proj=nn.Linear(emb_dim,2*width)
        self.conv2=nn.Conv2d(width,width,3,padding=1)
        nn.init.zeros_(self.conv2.weight);nn.init.zeros_(self.conv2.bias)
    def forward(self,x,emb):
        h=self.conv1(F.silu(self.norm1(x)))
        s,b=self.emb_proj(emb).chunk(2,dim=1)
        h=self.norm2(h)*(1+s[:,:,None,None])+b[:,:,None,None]
        h=self.conv2(F.silu(h))
        return (x+h)/math.sqrt(2)

class VelocityNet(nn.Module):
    def __init__(self,width=64,blocks=4,emb_dim=128):
        super().__init__()
        if width%8!=0: raise ValueError('width must be a multiple of 8')
        self.time_mlp=nn.Sequential(SinusoidalTimeEmbedding(emb_dim),nn.Linear(emb_dim,4*emb_dim),nn.SiLU(),nn.Linear(4*emb_dim,emb_dim))
        self.cond_mlp=nn.Sequential(nn.Linear(5,emb_dim),nn.SiLU(),nn.Linear(emb_dim,emb_dim))
        self.input_conv=nn.Conv2d(SHAPE[0],width,3,padding=1)
        self.blocks=nn.ModuleList(ResidualBlock(width,emb_dim) for _ in range(blocks))
        self.output_norm=nn.GroupNorm(8,width)
        self.output_conv=nn.Conv2d(width,SHAPE[0],3,padding=1)
        nn.init.zeros_(self.output_conv.weight);nn.init.zeros_(self.output_conv.bias)
    def forward(self,x,t,cond):
        if tuple(x.shape[1:])!=SHAPE: raise ValueError(f'Expected (B,{SHAPE}), got {tuple(x.shape)}')
        e=self.time_mlp(t)+self.cond_mlp(cond)
        h=self.input_conv(x)
        for block in self.blocks:h=block(h,e)
        return self.output_conv(F.silu(self.output_norm(h)))*geometry_mask(x.device,x.dtype)




def sample_gaussian(x):
    return torch.randn_like(x) * geometry_mask(x.device, x.dtype)

def fm_loss(model, x1, cond, generator=None):
    x0 = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=generator) * geometry_mask(x1.device, x1.dtype)
    t = torch.rand((len(x1),), device=x1.device, generator=generator)
    xt = x0 * (1 - t[:, None, None, None]) + x1 * t[:, None, None, None]
    return F.mse_loss(model(xt, t, cond), x1 - x0)

@torch.no_grad()
def integrate(model, x0, cond, steps=20, solver='euler'):
    if solver not in ('euler','heun'): raise ValueError(solver)
    if steps<1: raise ValueError('steps must be positive')
    x=x0;h=1.0/steps
    for k in range(steps):
        t=x.new_full((len(x),),k/steps)
        v=model(x,t,cond)
        if solver=='euler': x=x+h*v
        else:
            xp=x+h*v
            vn=model(xp,x.new_full((len(x),),(k+1)/steps),cond)
            x=x+(h/2)*(v+vn)
    return x
