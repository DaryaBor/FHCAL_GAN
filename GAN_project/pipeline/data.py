import json
import math
from dataclasses import dataclass

from itertools import cycle
from typing import Tuple, List, Optional, Union, Any, Sequence

import numpy as np
import torch
import torch.utils.data
import torchvision
from torch import nn
from torchvision import transforms

# ВАЖНО
"""
Датасеты могут быть двух типов:
1. Элемент - число или тензор. В этом случае элемент рассматривается как x в GAN
2. Элемент - tuple длины 2. В этом случае 1-ый элемент tuple - x, 2-й - y (условие)
y - либо число, либо тензор, либо tuple с числами/тензорами

Обёртка в виде UnifiedDatasetWrapper приводит оба типа датасетов ко 2-му (для датасета 1-го типа y = None).
Обёртку следует использовать пользователю.
"""


class RandomDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int, *args, **kwargs):
        sampler = torch.utils.data.sampler.RandomSampler(dataset, replacement=True)
        random_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size,
                                                               drop_last=False)

        super().__init__(dataset, batch_sampler=random_sampler, *args, **kwargs)


def get_random_infinite_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, *args, **kwargs):
    return cycle(RandomDataloader(dataset, batch_size=batch_size, *args, **kwargs))


def collate_fn(els_list: Sequence[Union[Tuple, int, torch.Tensor]]):
    if isinstance(els_list[0], tuple):
        return tuple(collate_fn(list(a)) for a in zip(*els_list))
    elif isinstance(els_list[0], int):
        return torch.Tensor(els_list)
    elif isinstance(els_list[0], torch.Tensor):
        return torch.stack(tuple(els_list))
    elif els_list[0] is None:
        return None
    else:
        raise RuntimeError


def stack_batches(batches_list):
    if isinstance(batches_list[0], tuple):
        return tuple(stack_batches(list(a)) for a in zip(*batches_list))
    elif isinstance(batches_list[0], torch.Tensor):
        return torch.concat(batches_list, dim=0)
    elif batches_list[0] is None:
        return None


def move_batch_to(batch, device):
    if isinstance(batch, tuple):
        return tuple(move_batch_to(subbatch, device) for subbatch in batch)
    elif batch is None:
        return None
    else:
        return batch.to(device)


class LinearTransform(nn.Module):
    def __init__(self, min_to: float, max_to: float, dim: int) -> None:
        super().__init__()
        self.min_to = min_to
        self.max_to = max_to
        self.dim = dim

    def forward(self, X) -> torch.Tensor:
        mins = X.min(dim=self.dim).values
        maxs = X.max(dim=self.dim).values

        coefs = (self.max_to - self.min_to) / (maxs - mins)
        biases = self.min_to - coefs * mins

        shape = list(X.shape)
        shape[self.dim] = 1

        coefs = coefs.reshape(*shape)
        biases = biases.reshape(*shape)

        y = coefs * X + biases
        y = torch.clip(y, self.min_to, self.max_to)  # clipping for making sure

        return y


class ExtractIndicesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices: Union[Tuple[int], int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, n: int):
        obj = self.dataset[n]
        if isinstance(self.indices, int):
            return obj[self.indices]
        else:
            return tuple(obj[i] for i in self.indices)



# USED
class UnifiedDatasetWrapper(torch.utils.data.Dataset):
    """
    Обёртка для поддержки датасетов обоих типов
    """
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset
        self.inverse_transform = getattr(dataset, 'inverse_transform', None)
        # У нас всегда есть inverse_transform, т.к. мы используем PhysicsDataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, n: int) -> Tuple[Any, Any]:
        element = self.dataset[n]
        if isinstance(element, tuple):
            assert len(element) == 2
            x, y = element
        else:
            # если есть только распределение энергии (у нас всегда есть ещё и точка, и импульс)
            x, y = element, None
        return x, y


def get_default_image_transform(dim: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        # Переводим цвета пикселей в отрезок [-1, 1] афинным преобразованием, изначально они в отрезке [0, 1]
        transforms.Normalize(tuple(0.5 for _ in range(dim)), tuple(0.5 for _ in range(dim)))
    ])


default_image_transform = get_default_image_transform(3)


def get_cifar_10_dataset(root='./cifar10', train: bool = True, keep_labels: bool = True, kept_labels: Optional[List[int]] = None):
    """
    :param kept_labels: images with which labels should be used
    """
    cifar_dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True,
                                                 transform=default_image_transform, )

    if kept_labels is not None:
        kept_indices = []
        for i in range(len(cifar_dataset)):
            if cifar_dataset[i][1] in kept_labels:
                kept_indices.append(i)

        cifar_dataset = torch.utils.data.Subset(cifar_dataset, kept_indices)

    if not keep_labels:
        cifar_dataset = ExtractIndicesDataset(cifar_dataset, indices=0)
    return cifar_dataset


def get_mnist_dataset(root='./mnist', train: bool = True, keep_labels: bool = True):
    mnist_dataset = torchvision.datasets.MNIST(root=root, train=train, download=True,
                                               transform=get_default_image_transform(1))

    if not keep_labels:
        mnist_dataset = ExtractIndicesDataset(mnist_dataset, indices=0)
    return mnist_dataset

# FHCal physical geometry, module energies and condition preparation.
@dataclass(frozen=True)
class ModuleGeometry:
    module_id: int
    x: float
    y: float
    z: float
    width: float
    height: float
    layers: int


class PhysicalGeometry(nn.Module):
    def __init__(self, modules, max_layers=10, cell_size=2.5, tolerance=1e-4):
        super().__init__()
        modules = sorted(modules, key=lambda m: m.module_id)
        if not modules or len({m.module_id for m in modules}) != len(modules):
            raise ValueError('Geometry must have distinct module IDs')
        if cell_size <= 0 or max_layers <= 0:
            raise ValueError('Invalid grid/layer size')
        for m in modules:
            if not all(math.isfinite(v) for v in (m.x, m.y, m.z, m.width, m.height)):
                raise ValueError('Nonfinite geometry')
            if min(m.width, m.height) <= 0 or not 1 <= m.layers <= max_layers:
                raise ValueError('Invalid module dimensions/layers')
        self.modules = tuple(modules)
        self.max_layers = max_layers
        self.cell_size = cell_size
        bounds = torch.tensor([[m.x-m.width/2, m.y-m.height/2,
                                m.x+m.width/2, m.y+m.height/2] for m in modules], dtype=torch.float64)
        xmin, ymin = bounds[:, :2].min(0).values.tolist()
        xmax, ymax = bounds[:, 2:].max(0).values.tolist()
        self.height = math.ceil((ymax-ymin)/cell_size)
        self.width = math.ceil((xmax-xmin)/cell_size)
        self.origin = (xmin, ymin)
        yy, xx = torch.meshgrid(torch.arange(self.height, dtype=torch.float64),
                                torch.arange(self.width, dtype=torch.float64), indexing='ij')
        x, y = xmin+xx.flatten()*cell_size, ymin+yy.flatten()*cell_size
        ox = (torch.minimum(bounds[:, 2, None], x+cell_size) - torch.maximum(bounds[:, 0, None], x)).clamp_min(0)
        oy = (torch.minimum(bounds[:, 3, None], y+cell_size) - torch.maximum(bounds[:, 1, None], y)).clamp_min(0)
        overlap = ox*oy
        sources, targets, edges = [], [], []
        for i, a in enumerate(modules):
            for j in range(i+1, len(modules)):
                b = modules[j]
                dx = min(bounds[i, 2], bounds[j, 2])-max(bounds[i, 0], bounds[j, 0])
                dy = min(bounds[i, 3], bounds[j, 3])-max(bounds[i, 1], bounds[j, 1])
                if dx > tolerance and dy > tolerance:
                    raise ValueError(f'Overlapping modules {a.module_id}, {b.module_id}')
                shared = float(dy) if abs(float(dx)) <= tolerance and dy > tolerance else (
                    float(dx) if abs(float(dy)) <= tolerance and dx > tolerance else 0.)
                if shared:
                    for s, t in ((i, j), (j, i)):
                        ms, mt = modules[s], modules[t]
                        ex, ey = mt.x-ms.x, mt.y-ms.y
                        sources.append(s); targets.append(t)
                        edges.append([ex/100, ey/100, math.hypot(ex, ey)/100, shared/20])
        mask = torch.tensor([[l < m.layers for l in range(max_layers)] for m in modules]).float()
        self.register_buffer('module_ids', torch.tensor([m.module_id for m in modules]))
        self.register_buffer('overlap', overlap.float())
        self.register_buffer('layer_mask', mask)
        self.register_buffer('virtual_mask', ((mask.T.double() @ overlap) > 0).float().reshape(max_layers, self.height, self.width))
        self.register_buffer('edge_index', torch.tensor([sources, targets], dtype=torch.long))
        self.register_buffer('edge_features', torch.tensor(edges, dtype=torch.float32).reshape(-1, 4))
        node = torch.tensor([[m.x/100, m.y/100, m.z/1000, m.width*m.height/400,
                              float(m.width >= 20), m.layers/max_layers] for m in modules])
        self.register_buffer('node_features', node)
        # Area-weighted static channels also support partially covered cells.
        weights = overlap / overlap.sum(0).clamp_min(1e-12)
        static = (node[:, 3:].double().T @ weights).float()
        coords = torch.stack(((x+cell_size/2)/100, (y+cell_size/2)/100)).float()
        active = (overlap.sum(0) > 0).float()[None]
        self.register_buffer('grid_features', torch.cat((coords, active, static)).reshape(6, self.height, self.width))

    def aggregate(self, density):
        if density.shape[1:] != (self.max_layers, self.height, self.width):
            raise ValueError('Expected density [B,L,H,W] matching geometry')
        return torch.einsum('mp,blp->bml', self.overlap, density.flatten(2)) * self.layer_mask


def fhcal_geometry_report_19():
    """Geometry and exact legacy map from fhcal_check_report.txt (muons/19).

    ROOT: 120000 events with stable IDs/centers; NPZ: 119999 nonzero events.
    fSectVec lengths: IDs 1..34 have 7 layers, IDs 35..54 have 10.
    Widths 15/20 cm and cm units retain the architecture assumptions; the report
    did not independently measure these sizes. This is a named configuration,
    not an inferred geometry for arbitrary NPZ files/runs.
    """
    z = 202.22000122070312
    # IDs 1..34: rows run from +Y to -Y, columns from +X to -X.
    centers = [(x, y)
               for y in (45., 30., 15., 0., -15., -30., -45.)
               for x in (30., 15., 0., -15., -30.)
               if (x, y) != (0., 0.)]
    # IDs 35..44 are on the right; IDs 45..54 are on the left.
    centers += [(x, y)
                for side_x in ((67.5, 47.5), (-47.5, -67.5))
                for y in (40., 20., 0., -20., -40.)
                for x in side_x]
    modules = []
    module_map = {}
    for mid, (x, y) in enumerate(centers, start=1):
        size = 15. if mid <= 34 else 20.
        layers = 7 if mid <= 34 else 10
        modules.append(ModuleGeometry(mid, x, y, z, size, size, layers))
        # Reproduce the checked NPZ writer, including floor, clipping and Y flip.
        # These are storage indices, not physical virtual-cell coordinates.
        col = min(8, max(0, math.floor((x + 67.5) / 15.)))
        row = 6 - min(6, max(0, math.floor((y + 45.) / (90. / 7.))))
        module_map[mid] = (row, col)
    return PhysicalGeometry(modules), module_map


def load_geometry(path):
    with open(path) as stream:
        config = json.load(stream)
    modules = [ModuleGeometry(**m) for m in config['modules']]
    if len(modules) != 54 or sum(m.width == m.height == 15 for m in modules) != 34 or sum(m.width == m.height == 20 for m in modules) != 20:
        raise ValueError('Expected 34 modules 15x15 cm and 20 modules 20x20 cm')
    return PhysicalGeometry(modules)


def geometry_from_root(path, module_metadata, tree_name='bmndata', branch_prefix='FHCalEvent/fModulesVec/fModulesVec'):
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
    """mapping_path accepts a JSON path or an explicitly selected ID -> (r,c) dict."""
    with np.load(path, allow_pickle=False) as data:
        energy = np.asarray(data['EnergyDeposit'], dtype=np.float32)
        point = np.asarray(data['ParticlePoint'], dtype=np.float32)
        momentum = np.asarray(data['ParticleMomentum'], dtype=np.float32)
        pdg = np.asarray(data['ParticlePDG']).reshape(-1)
        if energy.ndim == 4:
            if mapping_path is None:
                raise ValueError('Legacy NPZ requires module_map with verified module ID -> [row,col]')
            if isinstance(mapping_path, dict):
                module_map = mapping_path
            else:
                with open(mapping_path) as stream:
                    module_map = json.load(stream)
            energy = npz_grid_to_real_modules(energy, geometry, module_map)
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


class PhysicsDataset(torch.utils.data.Dataset):
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


def get_physics_dataset(path: str, train: bool = True, val_ratio: float = .1,
                        geometry=None, module_map=None) -> torch.utils.data.Dataset:
    """Physical energies [N,54,10]; conditions are standardized using train only."""
    if geometry is None:
        raise ValueError('Verified physical geometry is required for the virtual-grid model')
    return PhysicsDataset(path, geometry, module_map=module_map,
                          train=train, val_ratio=val_ratio)
