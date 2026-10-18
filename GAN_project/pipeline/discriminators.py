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


## класс с многоканальной архитектурой

class CaloganPhysicsDiscriminator3D(nn.Module):

    def __init__(
        self,
        act_func=F.leaky_relu,
        add_points_norms_and_angles: bool = True,
    ):
        super().__init__()

        self.activation = act_func
        self.add_points_norms_and_angles = add_points_norms_and_angles

        self.condition_dim = (
            7 if add_points_norms_and_angles else 5
        )

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
            indexing="ij",
        )

        occupied = central_mask + side_mask

        # =====================================================
        # 2. ДВУМЕРНЫЕ ПРИЗНАКИ
        #
        # Используются для расчёта физической энергии
        # и формирования геометрических 3D-признаков
        # =====================================================

        self.register_buffer(
            "central_mask",
            central_mask[None, None],
        )

        self.register_buffer(
            "side_mask",
            side_mask[None, None],
        )

        self.register_buffer(
            "central_square",
            (central_mask * 225.0)[None, None],
        )

        self.register_buffer(
            "side_square",
            (side_mask * 400.0)[None, None],
        )

        self.register_buffer(
            "coord_X",
            (xx * occupied)[None, None],
        )

        self.register_buffer(
            "coord_Y",
            (yy * occupied)[None, None],
        )

        # =====================================================
        # 3. ТРЁХМЕРНЫЕ МАСКИ
        #
        # (1, 1, 10, 11, 9)
        # =====================================================

        central_3d = (
            central_mask[None, None, None]
            .expand(1, 1, 10, 11, 9)
            .clone()
        )

        # Центральные модули: 7 секций
        central_3d[:, :, 7:, :, :] = 0.0

        side_3d = (
            side_mask[None, None, None]
            .expand(1, 1, 10, 11, 9)
            .clone()
        )

        self.register_buffer(
            "central_mask_3d",
            central_3d,
        )

        self.register_buffer(
            "side_mask_3d",
            side_3d,
        )

        self.register_buffer(
            "physical_mask_3d",
            central_3d + side_3d,
        )

        # =====================================================
        # 4. CONV3D
        #
        # 1 канал с энерговыделением
        # + 7 физических условий
        # + 6 геометрических признаков
        # = 14 входных каналов
        # =====================================================

        self.conv1 = spectral_norm(
            nn.Conv3d(
                in_channels=1 + self.condition_dim + 6,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        self.conv2 = spectral_norm(
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            )
        )

        self.conv3 = spectral_norm(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            )
        )

        self.conv4 = spectral_norm(
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        self.pool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )

        # =====================================================
        # 5. ENERGY-AWARE + УСЛОВИЯ
        #
        # 128 свёрточных признаков
        # + 7 физических условий
        # + 10 энергий продольных секций
        # + 1 полная энергия события
        # =====================================================

        self.fc1 = spectral_norm(
            nn.Linear(
                128 + self.condition_dim + 11,
                128,
            )
        )

        self.fc2 = spectral_norm(
            nn.Linear(
                128,
                1,
            )
        )

    # =========================================================
    # 6. ПОДГОТОВКА УСЛОВИЙ
    # =========================================================

    def _prepare_condition(self, y):

        point, momentum = y

        if self.add_points_norms_and_angles:

            angle = torch.atan2(
                point[:, 1],
                point[:, 0],
            ).unsqueeze(1)

            norm = torch.linalg.norm(
                point,
                dim=1,
                keepdim=True,
            )

            point = torch.cat(
                [point, angle, norm],
                dim=1,
            )

        condition = torch.cat(
            [momentum, point],
            dim=1,
        )

        return condition

    # =========================================================
    # 7. ГЕОМЕТРИЧЕСКИЕ ПРИЗНАКИ ДЛЯ CONV3D
    #
    # Всего 6 каналов, без канала глубины
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
            -1, -1, -1, -1,
        )

    # =========================================================
    # 8. FORWARD
    # =========================================================

    def forward(self, energy_map, y):

        # Вход: log1p(плотность)
        # (B, 10, 11, 9)

        if (
            energy_map.ndim != 4
            or tuple(energy_map.shape[1:]) != (10, 11, 9)
        ):
            raise ValueError(
                "Expected shape (B,10,11,9), "
                f"got {tuple(energy_map.shape)}"
            )

        batch_size = energy_map.size(0)

        condition = self._prepare_condition(y)

        # Передаём условия непосредственно в Conv3d
        condition_3d = (
            condition[:, :, None, None, None]
            .expand(
                -1,
                -1,
                self.depth,
                self.height,
                self.width,
            )
        )

        # -----------------------------------------------------
        # Свёрточная часть
        # -----------------------------------------------------

        x = torch.cat([
            energy_map.unsqueeze(1),
            condition_3d,
            self._geometry_features(batch_size),
        ], dim=1)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = self.pool(x)
        x = x.flatten(start_dim=1)

        # -----------------------------------------------------
        # ENERGY-AWARE
        #
        # Восстановление физической энергии
        # -----------------------------------------------------

        density = torch.expm1(
            energy_map
        ).clamp_min(0.0)

        area = (
            self.central_square
            + self.side_square
        )

        physical_energy = density * area

        # Энергия каждой продольной секции
        layer_energy = physical_energy.sum(
            dim=(2, 3)
        )

        # Полная энергия события
        total_energy = layer_energy.sum(
            dim=1,
            keepdim=True,
        )

        # Логарифмируем энергетические признаки
        energy_features = torch.cat([
            torch.log1p(layer_energy),
            torch.log1p(total_energy),
        ], dim=1)

        # -----------------------------------------------------
        # Объединяем информацию
        # -----------------------------------------------------

        x = torch.cat([
            x,
            condition,
            energy_features,
        ], dim=1)

        x = self.activation(self.fc1(x))

        # WGAN critic: без Sigmoid
        return self.fc2(x)
