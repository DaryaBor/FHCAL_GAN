from abc import abstractmethod
from typing import Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pipeline import _aux as aux


class Generator(nn.Module):
    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        y: Any = None
    ) -> torch.Tensor:
        """
        :param z: seed/noise for generation
        :param y: condition

        None means no condition.
        A generator knows the exact type of condition
        and how to use it for generation.

        If generator does not support conditions,
        it is expected to raise an exception.
        """
        pass


# ============================================================
# СТАРАЯ 2D-ВЕРСИЯ ГЕНЕРАТОРА
# ============================================================

class CaloganPhysicsGenerator(Generator):

    def __init__(
        self,
        noise_dim: int,
        act_func=F.relu,
        add_points_norms_and_angles: bool = True
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.activation = act_func

        self.add_points_norms_and_angles = (
            add_points_norms_and_angles
        )

        condition_dim = (
            7
            if add_points_norms_and_angles
            else 5
        )

        input_dim = (
            self.noise_dim
            + condition_dim
        )

        self.fc1 = nn.Linear(
            input_dim,
            256
        )

        self.bn1 = nn.BatchNorm1d(
            256
        )

        self.fc2 = nn.Linear(
            256 + condition_dim,
            512
        )

        self.bn2 = nn.BatchNorm1d(
            512
        )

        self.fc3 = nn.Linear(
            512 + condition_dim,
            1024
        )

        self.bn3 = nn.BatchNorm1d(
            1024
        )

        self.fc4 = nn.Linear(
            1024 + condition_dim,
            7 * 7 * 5
        )

    def _prepare_condition(
        self,
        y
    ):

        point, momentum = y

        if self.add_points_norms_and_angles:

            point = aux.add_angle_and_norm(
                point
            )

        condition = torch.cat(
            [
                momentum,
                point
            ],
            dim=1
        )

        return condition

    def forward(
        self,
        z: torch.Tensor,
        y
    ) -> torch.Tensor:

        condition = (
            self._prepare_condition(y)
        )

        x = torch.cat(
            [
                z,
                condition
            ],
            dim=1
        )

        x = self.activation(
            self.bn1(
                self.fc1(x)
            )
        )

        x = torch.cat(
            [
                x,
                condition
            ],
            dim=1
        )

        x = self.activation(
            self.bn2(
                self.fc2(x)
            )
        )

        x = torch.cat(
            [
                x,
                condition
            ],
            dim=1
        )

        x = self.activation(
            self.bn3(
                self.fc3(x)
            )
        )

        x = torch.cat(
            [
                x,
                condition
            ],
            dim=1
        )

        x = self.fc4(x)

        EnergyDeposit = x.view(
            -1,
            7,
            7,
            5
        )

        EnergyDeposit = F.relu(
            EnergyDeposit
        )

        return EnergyDeposit


# ============================================================
# CNN НА ВИРТУАЛЬНОЙ СЕТКЕ 2,5 СМ С АГРЕГАЦИЕЙ В ФИЗИЧЕСКИЕ МОДУЛИ
# ============================================================

class CaloganPhysicsGenerator3D(Generator):
    """CNN on a virtual 2.5 cm grid, integrated into physical module energies."""
    def __init__(self, geometry, condition_dim, noise_dim=64, hidden=64, condition_state=None):
        super().__init__()
        self.geometry, self.noise_dim = geometry, noise_dim
        # Persist the dataset encoder for inference in the existing GAN checkpoint.
        state = condition_state or {'mean': torch.zeros(6), 'scale': torch.ones(6),
                                    'pdg_codes': list(range(condition_dim-6))}
        self.register_buffer('condition_mean', state['mean'].clone())
        self.register_buffer('condition_scale', state['scale'].clone())
        self.register_buffer('pdg_codes', torch.tensor(state['pdg_codes'], dtype=torch.long))
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
