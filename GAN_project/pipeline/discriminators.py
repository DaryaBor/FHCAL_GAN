from abc import abstractmethod
from typing import Tuple, Any

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


## класс с 3d сверткой

class CaloganPhysicsDiscriminator3D(Discriminator):

    def __init__(
        self,
        act_func=F.leaky_relu,
        add_points_norms_and_angles: bool = True,
    ):
        super().__init__()

        self.activation = act_func
        self.add_points_norms_and_angles = add_points_norms_and_angles

    
        # (B, 1, 7, 7, 5)

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # (B, 1, 7, 7, 5)
       
        # (B, 32, 7, 7, 5)

        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=(1,2,2),
            padding=1,
        )
       
        # (B, 64, 7, 4, 3)

        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
        )
     
        # (B, 128, 7, 4, 3)

        self.conv4 = nn.Conv3d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # (B, 256, 7, 4, 3)

        # Сжимает весь трёхмерный объём до одного числа
        # в каждом из 256 каналов.
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        condition_dim = (
            7 if add_points_norms_and_angles else 5
        )

        self.fc1 = nn.Linear(
            256 + condition_dim,
            64,
        )
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(
        self,
        EnergyDeposit: torch.Tensor,
        y,
    ) -> torch.Tensor:

        point, momentum = y

        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(point)

        # Генератор и датасет возвращают:
        # (B, 7, 7, 5)
        #
        # Conv3d ожидает:
        # (B, channels, depth, height, width)
        #
        # Поэтому добавляем один канал:
        # (B, 7, 7, 5) → (B, 1, 7, 7, 5)

        if EnergyDeposit.ndim == 4:
            X = EnergyDeposit.unsqueeze(1)

        elif EnergyDeposit.ndim == 5:
            X = EnergyDeposit

        else:
            raise ValueError(
                'Ожидалась форма EnergyDeposit '
                '(batch, 7, 7, 5) или '
                '(batch, 1, 7, 7, 5), '
                f'но получена {tuple(EnergyDeposit.shape)}'
            )

        X = self.activation(
            self.conv1(X),
            negative_slope=0.2,
        )

        X = self.activation(
            self.conv2(X),
            negative_slope=0.2,
        )

        X = self.activation(
            self.conv3(X),
            negative_slope=0.2,
        )

        X = self.activation(
            self.conv4(X),
            negative_slope=0.2,
        )

     
        X = self.adaptive_pool(X)
        X = X.flatten(start_dim=1)

        condition = torch.cat(
            [momentum, point],
            dim=1,
        )
        
        X = torch.cat(
            [X, condition],
            dim=1,
        )

        X = F.leaky_relu(
            self.fc1(X),
            negative_slope=0.2,
        )

        X = F.leaky_relu(
            self.fc2(X),
            negative_slope=0.2,
        )
        return self.fc3(X)