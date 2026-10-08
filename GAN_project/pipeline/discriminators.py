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

        self.add_points_norms_and_angles = (
            add_points_norms_and_angles
        )

        self.condition_dim = (
            7 if add_points_norms_and_angles else 5
        )

        self.height = 11
        self.width = 9

        central_mask = torch.tensor([
            [0,0,1,1,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,1,1,0,0],
            [0,0,1,1,0,1,1,0,0],
            [0,0,1,1,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,1,1,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,1,1,0,0],
        ], dtype=torch.float32)

        side_mask = torch.tensor([
            [0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,1,1],
            [0,0,0,0,0,0,0,0,0],
        ], dtype=torch.float32)

        x_coords = [
            -67.5, -47.5,
            -30.0, -15.0, 0.0, 15.0, 30.0,
            47.5, 67.5,
        ]

        y_coords = [
            45.0,
            40.0,
            30.0,
            20.0,
            15.0,
            0.0,
            -15.0,
            -20.0,
            -30.0,
            -40.0,
            -45.0,
        ]

        coord_X = torch.zeros(
            self.height,
            self.width,
            dtype=torch.float32,
        )

        coord_Y = torch.zeros(
            self.height,
            self.width,
            dtype=torch.float32,
        )

        module_mask = (
            central_mask + side_mask
        )

        for row, y in enumerate(y_coords):
            for col, x in enumerate(x_coords):
                if module_mask[row, col] > 0:
                    coord_X[row, col] = x
                    coord_Y[row, col] = y

        central_square = (
            central_mask * 225.0
        )

        side_square = (
            side_mask * 400.0
        )

        central_depth = (
            central_mask * 7.0
        )

        side_depth = (
            side_mask * 10.0
        )

        self.register_buffer(
            "central_mask",
            central_mask.unsqueeze(0).unsqueeze(0),
        )

        self.register_buffer(
            "side_mask",
            side_mask.unsqueeze(0).unsqueeze(0),
        )

        self.register_buffer(
            "central_square",
            central_square.unsqueeze(0).unsqueeze(0),
        )

        self.register_buffer(
            "side_square",
            side_square.unsqueeze(0).unsqueeze(0),
        )

        self.register_buffer(
            "central_depth",
            central_depth.unsqueeze(0).unsqueeze(0),
        )

        self.register_buffer(
            "side_depth",
            side_depth.unsqueeze(0).unsqueeze(0),
        )

        self.register_buffer(
            "coord_X",
            coord_X.unsqueeze(0).unsqueeze(0),
        )

        self.register_buffer(
            "coord_Y",
            coord_Y.unsqueeze(0).unsqueeze(0),
        )

        self.conv1 = nn.Conv2d(
            in_channels=18,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc1 = nn.Linear(
            128
            + self.condition_dim
            + 11,
            128,
        )

        self.fc2 = nn.Linear(
            128,
            1,
        )

    def _prepare_condition(
        self,
        y,
    ):
        point, momentum = y

        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(
                point
            )

        condition = torch.cat(
            [
                momentum,
                point,
            ],
            dim=1,
        )

        return condition

    def forward(
        self,
        energy_map,
        y,
    ):

        condition = self._prepare_condition(y)

        batch_size = energy_map.size(0)

        central_mask = self.central_mask.expand(
            batch_size, -1, -1, -1
        )

        side_mask = self.side_mask.expand(
            batch_size, -1, -1, -1
        )

        central_square = self.central_square.expand(
            batch_size, -1, -1, -1
        )

        side_square = self.side_square.expand(
            batch_size, -1, -1, -1
        )

        central_depth = self.central_depth.expand(
            batch_size, -1, -1, -1
        )

        side_depth = self.side_depth.expand(
            batch_size, -1, -1, -1
        )

        coord_X = self.coord_X.expand(
            batch_size, -1, -1, -1
        )

        coord_Y = self.coord_Y.expand(
            batch_size, -1, -1, -1
        )

        x = torch.cat(
            [
                energy_map,
                central_mask,
                side_mask,
                central_square,
                side_square,
                central_depth,
                side_depth,
                coord_X,
                coord_Y,
            ],
            dim=1,
        )

        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.activation(x)

        x = self.pool(x)
        x = x.flatten(1)

        density = torch.expm1(
            energy_map
        ).clamp_min(0.0)

        area = (
            self.central_square
            + self.side_square
        )

        physical_energy = (
            density * area
        )

        layer_energy = physical_energy.sum(
            dim=(2, 3)
        )

        total_energy = layer_energy.sum(
            dim=1,
            keepdim=True,
        )

        log_layer_energy = torch.log1p(
            layer_energy
        )

        log_total_energy = torch.log1p(
            total_energy
        )

        energy_features = torch.cat(
            [
                log_layer_energy,
                log_total_energy,
            ],
            dim=1,
        )

        x = torch.cat(
            [
                x,
                condition,
                energy_features,
            ],
            dim=1,
        )

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)

        return x