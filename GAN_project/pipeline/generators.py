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
# ВЕРСИЯ С МНОГОКАНАЛЬНОЙ АРХИТЕКТУРОЙ
# ============================================================
class CaloganPhysicsGenerator3D(nn.Module):

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

        self.depth = 10
        self.height = 11
        self.width = 9

        # =====================================================
        # 1. ГЕОМЕТРИЯ FHCAL
        # =====================================================

        central_mask = torch.tensor([
            [0, 0, 1, 1, 1, 1, 1, 0, 0],  #  45
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #  40
            [0, 0, 1, 1, 1, 1, 1, 0, 0],  #  30
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  #  20
            [0, 0, 1, 1, 1, 1, 1, 0, 0],  #  15
            [0, 0, 1, 1, 0, 1, 1, 0, 0],  #   0
            [0, 0, 1, 1, 1, 1, 1, 0, 0],  # -15
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # -20
            [0, 0, 1, 1, 1, 1, 1, 0, 0],  # -30
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # -40
            [0, 0, 1, 1, 1, 1, 1, 0, 0],  # -45
        ], dtype=torch.float32)

        side_mask = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.float32)

        x_coords = torch.tensor([
            -67.5, -47.5, -30.0, -15.0, 0.0,
            15.0, 30.0, 47.5, 67.5,
        ])

        y_coords = torch.tensor([
            45.0, 40.0, 30.0, 20.0, 15.0, 0.0,
            -15.0, -20.0, -30.0, -40.0, -45.0,
        ])

        yy, xx = torch.meshgrid(
            y_coords,
            x_coords,
            indexing="ij"
        )

        occupied = central_mask + side_mask

        # =====================================================
        # 2. ДВУМЕРНЫЕ ПРИЗНАКИ
        #
        # Оставляем для совместимости с train.py
        # =====================================================

        self.register_buffer(
            "central_mask",
            central_mask[None, None]
        )

        self.register_buffer(
            "side_mask",
            side_mask[None, None]
        )

        self.register_buffer(
            "central_square",
            (central_mask * 225.0)[None, None]
        )

        self.register_buffer(
            "side_square",
            (side_mask * 400.0)[None, None]
        )

        self.register_buffer(
            "coord_X",
            (xx * occupied)[None, None]
        )

        self.register_buffer(
            "coord_Y",
            (yy * occupied)[None, None]
        )

        # =====================================================
        # 3. ТРЁХМЕРНЫЕ МАСКИ
        #
        # Размерность: (1, 1, 10, 11, 9)
        # =====================================================

        central_3d = (
            central_mask[None, None, None]
            .expand(1, 1, 10, 11, 9)
            .clone()
        )

        # Центральные модули имеют только 7 секций
        central_3d[:, :, 7:, :, :] = 0.0

        side_3d = (
            side_mask[None, None, None]
            .expand(1, 1, 10, 11, 9)
            .clone()
        )

        self.register_buffer(
            "central_mask_3d",
            central_3d
        )

        self.register_buffer(
            "side_mask_3d",
            side_3d
        )

        self.register_buffer(
            "physical_mask_3d",
            central_3d + side_3d
        )

        # =====================================================
        # 4. ПОЛНОСВЯЗНЫЕ СЛОИ
        #
        # Сохранены из multi-channel
        # =====================================================

        self.fc1 = nn.Linear(
            noise_dim + self.condition_dim,
            256
        )

        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(
            256 + self.condition_dim,
            512
        )

        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(
            512 + self.condition_dim,
            1024
        )

        self.bn3 = nn.BatchNorm1d(1024)

        # =====================================================
        # 5. СОЗДАНИЕ ТРЁХМЕРНОГО ОБЪЁМА
        #
        # (B, 16, 10, 11, 9)
        # =====================================================

        self.central_fc = nn.Linear(
            1024 + self.condition_dim,
            (
                self.hidden_channels
                * self.depth
                * self.height
                * self.width
            )
        )

        # =====================================================
        # 6. ТРЁХМЕРНЫЕ СВЁРТКИ
        #
        # 16 скрытых каналов
        # + 7 физических условий
        # + 6 геометрических признаков
        # = 29 входных каналов
        # =====================================================

        self.central_conv1 = nn.Conv3d(
            in_channels=(
                self.hidden_channels
                + self.condition_dim
                + 6
            ),
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.central_conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Два канала:
        # 0 - центральные модули
        # 1 - боковые модули
        self.energy_out = nn.Conv3d(
            in_channels=32,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0
        )

    # =========================================================
    # 7. ПОДГОТОВКА ФИЗИЧЕСКИХ УСЛОВИЙ
    # =========================================================

    def _prepare_condition(self, y):

        point, momentum = y

        if self.add_points_norms_and_angles:

            angle = torch.atan2(
                point[:, 1],
                point[:, 0]
            ).unsqueeze(1)

            norm = torch.linalg.norm(
                point,
                dim=1,
                keepdim=True
            )

            point = torch.cat(
                [point, angle, norm],
                dim=1
            )

        condition = torch.cat(
            [momentum, point],
            dim=1
        )

        return condition

    # =========================================================
    # 8. ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ ДЛЯ CONV3D
    # =========================================================

    def _geometry_features(self, batch_size):

        cm = self.central_mask_3d
        sm = self.side_mask_3d

        geometry = torch.cat([
            cm,
            sm,

            (
                self.central_square.unsqueeze(2)
                / 400.0
            ) * cm,

            (
                self.side_square.unsqueeze(2)
                / 400.0
            ) * sm,

            (
                self.coord_X.unsqueeze(2)
                / 67.5
            ) * self.physical_mask_3d,

            (
                self.coord_Y.unsqueeze(2)
                / 45.0
            ) * self.physical_mask_3d,

        ], dim=1)

        return geometry.expand(
            batch_size,
            -1, -1, -1, -1
        )

    # =========================================================
    # 9. FORWARD
    # =========================================================

    def forward(self, noise, y):

        condition = self._prepare_condition(y)

        batch_size = noise.size(0)

        # -----------------------------------------------------
        # Полносвязная часть
        # -----------------------------------------------------

        x = torch.cat(
            [noise, condition],
            dim=1
        )

        x = self.activation(
            self.bn1(self.fc1(x))
        )

        x = torch.cat(
            [x, condition],
            dim=1
        )

        x = self.activation(
            self.bn2(self.fc2(x))
        )

        x = torch.cat(
            [x, condition],
            dim=1
        )

        x = self.activation(
            self.bn3(self.fc3(x))
        )

        x = torch.cat(
            [x, condition],
            dim=1
        )

        # -----------------------------------------------------
        # Формирование трёхмерного объёма
        # -----------------------------------------------------

        x = self.central_fc(x)

        x = x.reshape(
            batch_size,
            self.hidden_channels,
            self.depth,
            self.height,
            self.width
        )

        # -----------------------------------------------------
        # Передача физических условий в Conv3d
        #
        # Аналогично test
        # -----------------------------------------------------

        condition_3d = (
            condition[:, :, None, None, None]
            .expand(
                -1,
                -1,
                self.depth,
                self.height,
                self.width
            )
        )

        geometry_3d = self._geometry_features(
            batch_size
        )

        x = torch.cat([
            x,
            condition_3d,
            geometry_3d
        ], dim=1)

        # -----------------------------------------------------
        # Трёхмерные свёртки
        # -----------------------------------------------------

        x = self.activation(
            self.central_conv1(x)
        )

        x = self.activation(
            self.central_conv2(x)
        )

        # -----------------------------------------------------
        # Выходная активация
        #
        # Softplus оставлен без изменений
        # -----------------------------------------------------

        density = F.softplus(
            self.energy_out(x)
        )

        log_density = torch.log1p(density)

        # -----------------------------------------------------
        # Центральные и боковые модули
        # -----------------------------------------------------

        central_density = (
            log_density[:, 0:1]
            * self.central_mask_3d
        )

        side_density = (
            log_density[:, 1:2]
            * self.side_mask_3d
        )

        # -----------------------------------------------------
        # Итоговая карта FHCal
        # -----------------------------------------------------

        physical_density = (
            central_density + side_density
        )

        physical_density = physical_density.squeeze(1)

        return physical_density
