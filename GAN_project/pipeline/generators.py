from abc import abstractmethod
from typing import Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pipeline import _aux as aux


class Generator(nn.Module):
    @abstractmethod
    def forward(self, z: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param z: seed/noise for generation
        :param y: condition
        None means no condition.
        A generator knows the exact type of condition and how to use it for generation.
        If generator does not support conditions, it is expected to raise an exception.
        """
        pass









class CaloganPhysicsGenerator(Generator):
    def __init__(self, noise_dim: int, act_func=F.relu, add_points_norms_and_angles: bool = True):
        super().__init__()
        self.noise_dim = noise_dim
        self.activation = act_func
        self.add_points_norms_and_angles = add_points_norms_and_angles

        condition_dim = 7 if add_points_norms_and_angles else 5
        input_dim = self.noise_dim + condition_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256 + condition_dim, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512 + condition_dim, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc4 = nn.Linear(1024 + condition_dim, 7 * 7 * 5)

    def _prepare_condition(self, y):
        point, momentum = y

        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(point)

        condition = torch.cat([momentum, point], dim=1)
        return condition

    def forward(self, z: torch.Tensor, y) -> torch.Tensor:
        condition = self._prepare_condition(y)

        x = torch.cat([z, condition], dim=1)

        x = self.activation(self.bn1(self.fc1(x)))

        x = torch.cat([x, condition], dim=1)
        x = self.activation(self.bn2(self.fc2(x)))

        x = torch.cat([x, condition], dim=1)
        x = self.activation(self.bn3(self.fc3(x)))

        x = torch.cat([x, condition], dim=1)
        x = self.fc4(x)

        EnergyDeposit = x.view(-1, 7, 7, 5)

        EnergyDeposit = F.relu(EnergyDeposit)

        return EnergyDeposit
    
# реализация с 3d сверткой
class CaloganPhysicsGenerator3D(Generator):

    def __init__(
        self,
        noise_dim: int,
        act_func=F.leaky_relu,
        add_points_norms_and_angles: bool = True,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.activation = act_func
        self.add_points_norms_and_angles = add_points_norms_and_angles

        self.condition_dim = (
            7 if add_points_norms_and_angles else 5
        )

        self.hidden_channels = 16
        self.depth = 7
        self.height = 7
        self.width = 5

        self.fc1 = nn.Linear(
            noise_dim + self.condition_dim,
            256,
        )
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(
            256 + self.condition_dim,
            512,
        )
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(
            512 + self.condition_dim,
            1024,
        )
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc4 = nn.Linear(
            1024 + self.condition_dim,
            self.hidden_channels
            * self.depth
            * self.height
            * self.width,
        )

        self.conv1 = nn.Conv3d(
            in_channels=self.hidden_channels + self.condition_dim,
            out_channels=self.hidden_channels,
            kernel_size=3,
            padding=1,
        )

        self.final_conv = nn.Conv3d(
            in_channels=self.hidden_channels,
            out_channels=1,
            kernel_size=1,
        )

    def forward(
        self,
        z: torch.Tensor,
        y,
    ) -> torch.Tensor:

        point, momentum = y

        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(point)

        condition = torch.cat(
            [momentum, point],
            dim=1,
        )

        x = torch.cat(
            [z, condition],
            dim=1,
        )

        x = self.activation(
            self.bn1(self.fc1(x))
        )

        x = torch.cat(
            [x, condition],
            dim=1,
        )
        x = self.activation(
            self.bn2(self.fc2(x))
        )

        x = torch.cat(
            [x, condition],
            dim=1,
        )
        x = self.activation(
            self.bn3(self.fc3(x))
        )

        x = torch.cat(
            [x, condition],
            dim=1,
        )
        x = self.fc4(x)

        x = x.view(
            -1,
            self.hidden_channels,
            self.depth,
            self.height,
            self.width,
        )

        condition_volume = condition.view(
            condition.shape[0],
            self.condition_dim,
            1,
            1,
            1,
        )

        condition_volume = condition_volume.expand(
            -1,
            -1,
            self.depth,
            self.height,
            self.width,
        )

        x = torch.cat(
            [x, condition_volume],
            dim=1,
        )

        x = self.activation(
            self.conv1(x)
        )

        x = self.final_conv(x)

        energy_deposit = F.softplus(x)

        return energy_deposit.squeeze(1)
    