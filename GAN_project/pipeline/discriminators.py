from abc import abstractmethod
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

    def forward(self, h, geometry):
        source, target = geometry.edge_index
        edge = geometry.edge_features.expand(h.size(0), -1, -1)
        message = self.message(torch.cat((h[:, source], h[:, target], edge), -1))
        aggregate = torch.zeros_like(h).index_add(1, target, message)
        degree = torch.bincount(target, minlength=h.size(1)).clamp_min(1).to(h.dtype)
        aggregate = aggregate/degree[None, :, None]
        return self.norm(h+self.update(torch.cat((h, aggregate), -1)))


class CaloganPhysicsDiscriminator3D(Discriminator):
    """Conditional GNN critic on physical modules, without sigmoid."""
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
