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
        module_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.activation = act_func
        self.add_points_norms_and_angles = (
            add_points_norms_and_angles
        )

        self.condition_dim = (
            7 if add_points_norms_and_angles else 5
        )

        self.hidden_channels = 16
        self.context_dim = 32

        # -----------------------------------------------------
        # Геометрия
        # -----------------------------------------------------

        # Центральные модули имеют 7 продольных секций
        self.central_depth = 7

        # Большие боковые модули имеют 10 продольных секций
        self.side_depth = 10

        # Количество строк по вертикали
        self.height = 7

        # Центральная область содержит 5 столбцов
        self.central_width = 5

        # Каждая боковая область содержит 2 столбца
        self.side_width = 2

        # -----------------------------------------------------
        # Общая полносвязная часть
        # -----------------------------------------------------

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

        # =====================================================
        # Центральная область
        # Форма: 7 продольных слоёв × 7 строк × 5 столбцов
        # =====================================================

        self.central_fc = nn.Linear(
            1024 + self.condition_dim,
            (
                self.hidden_channels
                * self.central_depth
                * self.height
                * self.central_width
            ),
        )

        self.central_conv1 = nn.Conv3d(
            in_channels=(
                self.hidden_channels
                + self.condition_dim
            ),
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.central_conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.central_out = nn.Conv3d(
            in_channels=self.hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # =====================================================
        # Контексты от границ центральной области
        # =====================================================

        self.left_context_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=self.context_dim,
                kernel_size=(3, 3, 1),
                stride=1,
                padding=(1, 1, 0),
            ),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.right_context_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_channels,
                out_channels=self.context_dim,
                kernel_size=(3, 3, 1),
                stride=1,
                padding=(1, 1, 0),
            ),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        side_input_dim = (
            noise_dim
            + self.condition_dim
            + self.context_dim
        )

        side_conv_input_channels = (
            self.hidden_channels
            + self.condition_dim
            + self.context_dim
        )

        # =====================================================
        # Левая боковая область
        # Форма: 10 продольных слоёв × 7 строк × 2 столбца
        # =====================================================

        self.left_fc = nn.Linear(
            side_input_dim,
            (
                self.hidden_channels
                * self.side_depth
                * self.height
                * self.side_width
            ),
        )

        # Сначала каждый из двух столбцов обрабатывается отдельно
        self.left_conv1 = nn.Conv3d(
            in_channels=side_conv_input_channels,
            out_channels=32,
            kernel_size=(3, 3, 1),
            stride=1,
            padding=(1, 1, 0),
        )

        # Затем учитывается связь между двумя столбцами
        self.left_conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.left_out = nn.Conv3d(
            in_channels=self.hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # =====================================================
        # Правая боковая область
        # Форма: 10 продольных слоёв × 7 строк × 2 столбца
        # =====================================================

        self.right_fc = nn.Linear(
            side_input_dim,
            (
                self.hidden_channels
                * self.side_depth
                * self.height
                * self.side_width
            ),
        )

        self.right_conv1 = nn.Conv3d(
            in_channels=side_conv_input_channels,
            out_channels=32,
            kernel_size=(3, 3, 1),
            stride=1,
            padding=(1, 1, 0),
        )

        self.right_conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=self.hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.right_out = nn.Conv3d(
            in_channels=self.hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # =====================================================
        # Маска существующих модулей
        # =====================================================

        if module_mask is None:
            module_mask = torch.ones(
                7,
                9,
                dtype=torch.float32,
            )
        else:
            module_mask = torch.as_tensor(
                module_mask,
                dtype=torch.float32,
            )

            if tuple(module_mask.shape) != (7, 9):
                raise ValueError(
                    "module_mask должна иметь форму (7, 9), "
                    f"получена {tuple(module_mask.shape)}"
                )

        # Выход генератора:
        # (B, 7, 9, 10)
        #
        # Маска:
        # (1, 7, 9, 1)
        self.register_buffer(
            "module_mask",
            module_mask.view(1, 7, 9, 1),
        )

    def _prepare_condition(
        self,
        y,
    ) -> torch.Tensor:

        point, momentum = y

        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(point)

        condition = torch.cat(
            [momentum, point],
            dim=1,
        )

        return condition

    @staticmethod
    def _expand_condition(
        condition: torch.Tensor,
        depth: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Преобразует condition:

        (B, C)
            ↓
        (B, C, depth, height, width)
        """

        return condition.view(
            condition.shape[0],
            condition.shape[1],
            1,
            1,
            1,
        ).expand(
            -1,
            -1,
            depth,
            height,
            width,
        )

    def forward(
        self,
        z: torch.Tensor,
        y,
    ) -> torch.Tensor:

        condition = self._prepare_condition(y)

        # =====================================================
        # 1. Общее скрытое представление
        # =====================================================

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

        central_input = torch.cat(
            [x, condition],
            dim=1,
        )

        # =====================================================
        # 2. Центральная область глубиной 7
        # =====================================================

        central = self.central_fc(
            central_input
        )

        central = central.view(
            -1,
            self.hidden_channels,
            self.central_depth,
            self.height,
            self.central_width,
        )

        # central:
        # (B, 16, 7, 7, 5)

        central_condition = self._expand_condition(
            condition=condition,
            depth=self.central_depth,
            height=self.height,
            width=self.central_width,
        )

        central = torch.cat(
            [
                central,
                central_condition,
            ],
            dim=1,
        )

        central = self.activation(
            self.central_conv1(central)
        )

        central_features = self.activation(
            self.central_conv2(central)
        )

        # central_features:
        # (B, 16, 7, 7, 5)

        central_energy = F.softplus(
            self.central_out(central_features),
            beta=5.0
        )

        # central_energy:
        # (B, 1, 7, 7, 5)

        # =====================================================
        # 3. Контекст левой и правой границы центра
        # =====================================================

        left_boundary = central_features[..., :1]

        # (B, 16, 7, 7, 1)

        right_boundary = central_features[..., -1:]

        # (B, 16, 7, 7, 1)

        left_context = self.left_context_encoder(
            left_boundary
        ).flatten(start_dim=1)

        right_context = self.right_context_encoder(
            right_boundary
        ).flatten(start_dim=1)

        # left_context:
        # (B, 32)
        #
        # right_context:
        # (B, 32)

        # =====================================================
        # 4. Левая боковая область глубиной 10
        # =====================================================

        left_input = torch.cat(
            [
                z,
                condition,
                left_context,
            ],
            dim=1,
        )

        left = self.left_fc(
            left_input
        )

        left = left.view(
            -1,
            self.hidden_channels,
            self.side_depth,
            self.height,
            self.side_width,
        )

        # left:
        # (B, 16, 10, 7, 2)

        left_condition = torch.cat(
            [
                condition,
                left_context,
            ],
            dim=1,
        )

        left_condition_volume = self._expand_condition(
            condition=left_condition,
            depth=self.side_depth,
            height=self.height,
            width=self.side_width,
        )

        left = torch.cat(
            [
                left,
                left_condition_volume,
            ],
            dim=1,
        )

        left = self.activation(
            self.left_conv1(left)
        )

        # (B, 32, 10, 7, 2)

        left = self.activation(
            self.left_conv2(left)
        )

        # (B, 16, 10, 7, 2)

        left_energy = F.softplus(
            self.left_out(left),
            beta=5.0
        )

        # left_energy:
        # (B, 1, 10, 7, 2)

        # =====================================================
        # 5. Правая боковая область глубиной 10
        # =====================================================

        right_input = torch.cat(
            [
                z,
                condition,
                right_context,
            ],
            dim=1,
        )

        right = self.right_fc(
            right_input
        )

        right = right.view(
            -1,
            self.hidden_channels,
            self.side_depth,
            self.height,
            self.side_width,
        )

        # right:
        # (B, 16, 10, 7, 2)

        right_condition = torch.cat(
            [
                condition,
                right_context,
            ],
            dim=1,
        )

        right_condition_volume = self._expand_condition(
            condition=right_condition,
            depth=self.side_depth,
            height=self.height,
            width=self.side_width,
        )

        right = torch.cat(
            [
                right,
                right_condition_volume,
            ],
            dim=1,
        )

        right = self.activation(
            self.right_conv1(right)
        )

        # (B, 32, 10, 7, 2)

        right = self.activation(
            self.right_conv2(right)
        )

        # (B, 16, 10, 7, 2)

        right_energy = F.softplus(
            self.right_out(right),
            beta=5.0
        )

        # right_energy:
        # (B, 1, 10, 7, 2)

        # =====================================================
        # 6. Дополнение центральной части до глубины 10
        # =====================================================

        depth_padding = (
            self.side_depth
            - self.central_depth
        )

        # Добавляем нули только после 7-го продольного слоя.
        #
        # Для 5D-тензора порядок padding:
        # (width_left, width_right,
        #  height_left, height_right,
        #  depth_left, depth_right)
        central_energy_padded = F.pad(
            central_energy,
            pad=(
                0,
                0,
                0,
                0,
                0,
                depth_padding,
            ),
            mode="constant",
            value=0.0,
        )

        # central_energy_padded:
        # (B, 1, 10, 7, 5)
        #
        # Слои 1–7 содержат результат генератора.
        # Слои 8–10 равны нулю.

        # =====================================================
        # 7. Сборка полного FHCal
        # =====================================================

        full_energy = torch.cat(
            [
                left_energy,
                central_energy_padded,
                right_energy,
            ],
            dim=4,
        )

        # Ширина:
        # 2 + 5 + 2 = 9
        #
        # full_energy:
        # (B, 1, 10, 7, 9)

        full_energy = full_energy.squeeze(1)

        # (B, 10, 7, 9)

        full_energy = full_energy.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        # Итог:
        # (B, 7, 9, 10)

        full_energy = (
            full_energy
            * self.module_mask
        )

        full_energy = full_energy.permute(
            0,
            3,
            1,
            2,
        ).contiguous()
        return full_energy
    