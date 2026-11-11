"""Conditional CNN -> physical overlap integration -> edge-aware GNN.

Lengths are cm; generator outputs energy / cm², critic inputs physical energy.
Geometry is explicit: no inference from the legacy 7x9 storage indices.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


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


class VirtualCNNGenerator(nn.Module):
    def __init__(self, geometry, condition_dim, noise_dim=64, hidden=64):
        super().__init__()
        self.geometry, self.noise_dim = geometry, noise_dim
        self.condition = nn.Sequential(nn.Linear(condition_dim, hidden), nn.SiLU())
        self.seed = nn.Linear(noise_dim+hidden, hidden*4*4)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1),
                                                 nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1)) for _ in range(3)])
        self.output = nn.Sequential(nn.Conv2d(hidden+6, hidden, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(hidden, geometry.max_layers, 1), nn.Softplus())
        self.hidden = hidden

    def virtual_density(self, noise, conditions):
        h = self.seed(torch.cat((noise, self.condition(conditions)), -1)).reshape(-1, self.hidden, 4, 4)
        for block in self.blocks:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
            h = F.silu(h+block(h))
        h = F.interpolate(h, size=(self.geometry.height, self.geometry.width), mode='bilinear', align_corners=False)
        static = self.geometry.grid_features.expand(h.size(0), -1, -1, -1)
        return self.output(torch.cat((h, static), 1))*self.geometry.virtual_mask

    def forward(self, noise, conditions):
        return self.geometry.aggregate(self.virtual_density(noise, conditions))


class MessageBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(hidden*2+4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, geometry):
        source, target = geometry.edge_index
        edge = geometry.edge_features.expand(h.size(0), -1, -1)
        message = self.message(torch.cat((h[:, source], h[:, target], edge), -1))
        aggregate = torch.zeros_like(h).index_add(1, target, message)
        degree = torch.bincount(target, minlength=h.size(1)).clamp_min(1).to(h.dtype)
        aggregate = aggregate/degree[None, :, None]
        return self.norm(h+self.update(torch.cat((h, aggregate), -1)))


class GraphCritic(nn.Module):
    def __init__(self, geometry, condition_dim, hidden=64, depth=4):
        super().__init__()
        self.geometry = geometry
        self.condition = nn.Sequential(nn.Linear(condition_dim, hidden), nn.SiLU())
        self.input = nn.Linear(geometry.max_layers*2+1+6+hidden, hidden)
        self.blocks = nn.ModuleList([MessageBlock(hidden) for _ in range(depth)])
        self.output = nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, energy, conditions):
        energy = energy*self.geometry.layer_mask
        b, m, _ = energy.shape
        h = self.input(torch.cat((energy, torch.log1p(energy.clamp_min(0)), energy.sum(-1, keepdim=True),
                                  self.geometry.node_features.expand(b, -1, -1),
                                  self.condition(conditions)[:, None].expand(-1, m, -1)), -1))
        for block in self.blocks:
            h = block(h, self.geometry)
        return self.output(torch.cat((h.mean(1), h.sum(1)), -1)).squeeze(-1)


def gradient_penalty(critic, real, fake, conditions):
    epsilon = torch.rand(real.size(0), 1, 1, device=real.device)
    interpolated = (epsilon*real+(1-epsilon)*fake.detach()).requires_grad_(True)
    score = critic(interpolated, conditions)
    gradient = torch.autograd.grad(score.sum(), interpolated, create_graph=True)[0]
    gradient = gradient*critic.geometry.layer_mask
    return (gradient.flatten(1).norm(2, dim=1)-1).square().mean()


def critic_loss(critic, real, fake, conditions, gp_weight=10.):
    return critic(fake.detach(), conditions).mean()-critic(real, conditions).mean()+gp_weight*gradient_penalty(critic, real, fake, conditions)


def generator_loss(critic, fake, conditions):
    return -critic(fake, conditions).mean()
