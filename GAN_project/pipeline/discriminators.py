from abc import abstractmethod
import math
from typing import Tuple, Any
from torch.nn.utils import spectral_norm
import torch
import torch.nn.functional as F
from torch import nn

from pipeline import _aux as aux


class Discriminator(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param x: object from the considered space
        :param y: condition
        None means no condition.
        A discriminator knows the exact type of condition and how to use it.
        If discriminator does not support conditions, it is expected to raise an exception.
        """
        pass


def save_dimensions_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    works only for odd kernel size values
    returns padding size such that the output has the same coordinate dimensions
    """
    res = []
    for sz in kernel_size:
        if sz % 2 == 0:
            raise ValueError('Only odd kernel size values are supported')
        res.append((sz - 1) // 2)
    return tuple(res)



class CaloganPhysicsDiscriminator(Discriminator):
    def __init__(self, act_func=F.leaky_relu, add_points_norms_and_angles: bool = True):
        super().__init__()
        self.activation = act_func
        self.add_points_norms_and_angles = add_points_norms_and_angles

        # Свертки с stride=2 для уменьшения размера
        self.conv1 = nn.Conv2d(7, 32, 3, stride=2, padding=1)  # 7x9 -> 4x5
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 4x5 -> 2x3
        
        # Дополнительные свертки без уменьшения размера
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # 2x3 -> 2x3
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)  # 2x3 -> 2x3
        
        # Adaptive pooling для получения 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        condition_dim = 7 if add_points_norms_and_angles else 5
        self.fc1 = nn.Linear(256 + condition_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, EnergyDeposit, y):
        point, momentum = y
        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(point)
        
        X = self.activation(self.conv1(EnergyDeposit))
       
        
        X = self.activation(self.conv2(X))
       
        
        X = self.activation(self.conv3(X))
       
        
        X = self.activation(self.conv4(X))
        
        
        X = self.adaptive_pool(X)
     
        
        X = X.reshape(-1, 256)
        X = torch.cat([X, momentum, point], dim=1)
        
        X = F.leaky_relu(self.fc1(X))
        X = F.leaky_relu(self.fc2(X))
        
        return self.fc3(X)


# GNN-критик по физическим модулям; прежнее имя класса сохранено.

class MessageBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(hidden*2+4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edge_index, edge_features):
        source, target = edge_index
        message = self.message(torch.cat((h[source], h[target], edge_features), -1))
        aggregate = torch.zeros_like(h).index_add(0, target, message)
        degree = torch.bincount(target, minlength=h.size(0)).clamp_min(1).to(h.dtype)
        aggregate = aggregate/degree[:, None]
        return self.norm(h+self.update(torch.cat((h, aggregate), -1)))


class CaloganPhysicsDiscriminator3D(Discriminator):
    """Conditional radius GNN on the active physical modules of each event."""
    def __init__(self, geometry, condition_dim, hidden=64, depth=4,
                 neighbor_radius=25.0, active_energy_threshold=1e-6):
        super().__init__()
        if not math.isfinite(neighbor_radius) or neighbor_radius <= 0:
            raise ValueError('neighbor_radius must be finite and positive (cm)')
        if not math.isfinite(active_energy_threshold) or active_energy_threshold < 0:
            raise ValueError('active_energy_threshold must be finite and nonnegative')
        self.geometry = geometry
        self.neighbor_radius = float(neighbor_radius)
        self.active_energy_threshold = float(active_energy_threshold)
        xy = geometry.node_features[:, :2] * 100
        delta = xy[None, :, :] - xy[:, None, :]
        distance = delta.norm(dim=-1)
        neighbors = (distance > 0) & (distance <= self.neighbor_radius + 1e-4)
        source, target = neighbors.nonzero(as_tuple=True)
        shared = torch.zeros_like(distance)
        shared[geometry.edge_index[0], geometry.edge_index[1]] = geometry.edge_features[:, 3]
        edge_features = torch.cat((delta[source, target]/100,
                                   distance[source, target, None]/100,
                                   shared[source, target, None]), -1)
        # Derived from geometry/config; keep the existing checkpoint tensor keys.
        self.register_buffer('radius_edge_index', torch.stack((source, target)), persistent=False)
        self.register_buffer('radius_edge_features', edge_features, persistent=False)
        self.condition = nn.Sequential(nn.Linear(condition_dim, hidden), nn.SiLU())
        self.input = nn.Linear(geometry.max_layers*2+1+6+hidden, hidden)
        self.blocks = nn.ModuleList([MessageBlock(hidden) for _ in range(depth)])
        self.output = nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, energy, conditions):
        energy = energy*self.geometry.layer_mask
        b, m, _ = energy.shape
        active = energy.sum(-1) > self.active_energy_threshold
        batch_index, module_index = active.nonzero(as_tuple=True)
        active_energy = energy[batch_index, module_index]
        h = self.input(torch.cat((active_energy, torch.log1p(active_energy.clamp_min(0)),
                                  active_energy.sum(-1, keepdim=True),
                                  self.geometry.node_features[module_index],
                                  self.condition(conditions)[batch_index]), -1))

        node_index = torch.full_like(active, -1, dtype=torch.long)
        node_index[batch_index, module_index] = torch.arange(h.size(0), device=energy.device)
        source, target = self.radius_edge_index
        edge_batch, edge_id = (active[:, source] & active[:, target]).nonzero(as_tuple=True)
        edge_index = torch.stack((node_index[edge_batch, source[edge_id]],
                                  node_index[edge_batch, target[edge_id]]))
        edge_features = self.radius_edge_features[edge_id]
        for block in self.blocks:
            h = block(h, edge_index, edge_features)
        total = h.new_zeros(b, h.size(-1)).index_add(0, batch_index, h)
        count = active.sum(1, keepdim=True).clamp_min(1).to(h.dtype)
        return self.output(torch.cat((total/count, total), -1)).squeeze(-1)
