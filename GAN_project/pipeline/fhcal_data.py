"""Validated physical geometry, explicit NPZ mapping and train-only conditions."""
import json

import numpy as np
import torch
from .fhcal_virtual25_gnn import ModuleGeometry, PhysicalGeometry


def load_geometry(path):
    with open(path) as stream:
        config = json.load(stream)
    modules = [ModuleGeometry(**m) for m in config['modules']]
    if len(modules) != 54 or sum(m.width == m.height == 15 for m in modules) != 34 or sum(m.width == m.height == 20 for m in modules) != 20:
        raise ValueError('Expected 34 modules 15x15 cm and 20 modules 20x20 cm')
    return PhysicalGeometry(modules)


def geometry_from_root(path, module_metadata, tree_name='bmndata', branch_prefix='FHCalEvent.fModulesVec'):
    """Read centers/IDs; sizes and VERIFIED fSectVec layer counts are required.

    module_metadata: {real_id: {'width': ..., 'height': ..., 'layers': ...}}.
    No assumptions about fZ being the front face. Requires optional uproot.
    Branch prefix is explicit because ROOT split-branch names vary by run.
    """
    import uproot
    with uproot.open(path) as root:
        tree = root[tree_name]
        fields = {name: tree[f'{branch_prefix}.{name}'].array(entry_stop=1, library='np')[0]
                  for name in ('fModId', 'fX', 'fY', 'fZ')}
    ids = [int(i) for i in fields['fModId']]
    if len(ids) != len(set(ids)) or set(ids) != set(module_metadata):
        raise ValueError('ROOT IDs and verified metadata must match exactly')
    return [ModuleGeometry(i, float(x), float(y), float(z), **module_metadata[i])
            for i, x, y, z in zip(ids, fields['fX'], fields['fY'], fields['fZ'])]


def npz_grid_to_real_modules(energy, geometry, module_to_grid):
    """No inferred historical mapping; reject collisions and lost energy."""
    if energy.ndim != 4 or energy.shape[1:] != (7, 9, geometry.max_layers):
        raise ValueError('Legacy EnergyDeposit must be [N,7,9,10]')
    ids = [m.module_id for m in geometry.modules]
    mapping = {int(k): tuple(v) for k, v in module_to_grid.items()}
    if set(mapping) != set(ids) or len(set(mapping.values())) != len(ids):
        raise ValueError('Mapping must cover every module once without collisions')
    if any(len(v) != 2 or any(not isinstance(i, int) for i in v) or not 0 <= v[0] < 7 or not 0 <= v[1] < 9 for v in mapping.values()):
        raise ValueError('Invalid mapping coordinates')
    occupied = np.zeros((7, 9), dtype=bool)
    for r, c in mapping.values():
        occupied[r, c] = True
    if np.any(energy[:, ~occupied, :] != 0):
        raise ValueError('Unmapped grid slots contain energy')
    result = np.stack([energy[:, mapping[i][0], mapping[i][1], :] for i in ids], 1)
    if not np.allclose(energy.sum((1, 2, 3)), result.sum((1, 2)), rtol=1e-5, atol=1e-6):
        raise ValueError('Conversion did not conserve energy')
    return result


def load_events(path, geometry, mapping_path=None):
    with np.load(path, allow_pickle=False) as data:
        energy = np.asarray(data['EnergyDeposit'], dtype=np.float32)
        point = np.asarray(data['ParticlePoint'], dtype=np.float32)
        momentum = np.asarray(data['ParticleMomentum'], dtype=np.float32)
        pdg = np.asarray(data['ParticlePDG']).reshape(-1)
        if energy.ndim == 4:
            if mapping_path is None:
                raise ValueError('Legacy NPZ requires --module-map with verified module ID -> [row,col]')
            with open(mapping_path) as stream:
                energy = npz_grid_to_real_modules(energy, geometry, json.load(stream))
        else:
            # Native physical format must carry its ordering explicitly.
            ids = list(np.asarray(data['ModuleIds']).tolist())
            expected = [m.module_id for m in geometry.modules]
            if len(set(ids)) != len(ids) or set(ids) != set(expected):
                raise ValueError('ModuleIds disagree with geometry')
            energy = energy[:, [ids.index(i) for i in expected]]
    n = len(energy)
    if energy.shape != (n, len(geometry.modules), geometry.max_layers) or point.shape != (n, 3) or momentum.shape != (n, 3) or pdg.shape != (n,):
        raise ValueError('Inconsistent NPZ event dimensions')
    if not all(np.isfinite(a).all() for a in (energy, point, momentum, pdg)) or np.any(energy < 0) or np.any(pdg != np.floor(pdg)):
        raise ValueError('Invalid energy/condition values')
    if np.any(energy[:, geometry.layer_mask.numpy() == 0] != 0):
        raise ValueError('Nonzero energy in absent layers: verify fSectVec and NPZ')
    return torch.from_numpy(energy), torch.from_numpy(np.concatenate((point, momentum), 1)), torch.from_numpy(pdg.astype(np.int64))


class ConditionEncoder:
    """c = standardized (x0,y0,z0,px,py,pz) + PDG one-hot.

    Fit once on train; checkpoint statistics/vocabulary for validation/inference.
    """
    def __init__(self, values, pdg):
        self.mean = values.mean(0)
        self.scale = values.std(0, unbiased=False).clamp_min(1e-6)
        self.pdg_codes = sorted(int(i) for i in pdg.unique())

    def transform(self, values, pdg):
        codes = torch.tensor(self.pdg_codes, device=pdg.device)
        one_hot = (pdg[:, None] == codes[None]).float()
        if not (one_hot.sum(1) == 1).all():
            raise ValueError('PDG not represented in training split')
        return torch.cat(((values-self.mean.to(values.device))/self.scale.to(values.device), one_hot), 1)

    def state_dict(self):
        return {'mean': self.mean, 'scale': self.scale, 'pdg_codes': self.pdg_codes}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls.__new__(cls)
        obj.mean, obj.scale, obj.pdg_codes = state['mean'], state['scale'], state['pdg_codes']
        return obj


class PhysicalModulesDataset(torch.utils.data.Dataset):
    """Same (x,y) dataset contract as PhysicsDataset, with physical energies."""
    def __init__(self, path, geometry, module_map=None, train=True, val_ratio=.1):
        energy, values, pdg = load_events(path, geometry, module_map)
        if len(energy) < 2 or not 0 < val_ratio < 1:
            raise ValueError('Need at least two events and 0 < val_ratio < 1')
        # Preserve the existing PhysicsDataset split without modifying global RNG.
        rng = np.random.RandomState(0x3df3fa)
        count = min(len(energy)-1, max(1, int(len(energy)*val_ratio)))
        val_indices = rng.choice(np.arange(len(energy)), size=count, replace=False)
        val_mask = np.zeros(len(energy), dtype=bool)
        val_mask[val_indices] = True
        train_indices = np.arange(len(energy))[~val_mask]
        encoder = ConditionEncoder(values[train_indices], pdg[train_indices])
        indices = train_indices if train else val_indices
        self.energy = energy[indices]
        self.conditions = encoder.transform(values[indices], pdg[indices])
        self.condition_dim = 6 + len(encoder.pdg_codes)
        self.condition_state = encoder.state_dict()
        self.inverse_transform = None  # Already in original energy units.

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, index):
        return self.energy[index], self.conditions[index]
