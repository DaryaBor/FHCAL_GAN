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

    

        self.height = 11
        self.width = 9

        module_coords = {
            1:  [30.0, 45.0, 202.22000122070312],
            2:  [15.0, 45.0, 202.22000122070312],
            3:  [0.0, 45.0, 202.22000122070312],
            4:  [-15.0, 45.0, 202.22000122070312],
            5:  [-30.0, 45.0, 202.22000122070312],

            6:  [30.0, 30.0, 202.22000122070312],
            7:  [15.0, 30.0, 202.22000122070312],
            8:  [0.0, 30.0, 202.22000122070312],
            9:  [-15.0, 30.0, 202.22000122070312],
            10: [-30.0, 30.0, 202.22000122070312],

            11: [30.0, 15.0, 202.22000122070312],
            12: [15.0, 15.0, 202.22000122070312],
            13: [0.0, 15.0, 202.22000122070312],
            14: [-15.0, 15.0, 202.22000122070312],
            15: [-30.0, 15.0, 202.22000122070312],

            16: [30.0, 0.0, 202.22000122070312],
            17: [15.0, 0.0, 202.22000122070312],
            18: [-15.0, 0.0, 202.22000122070312],
            19: [-30.0, 0.0, 202.22000122070312],

            20: [30.0, -15.0, 202.22000122070312],
            21: [15.0, -15.0, 202.22000122070312],
            22: [0.0, -15.0, 202.22000122070312],
            23: [-15.0, -15.0, 202.22000122070312],
            24: [-30.0, -15.0, 202.22000122070312],

            25: [30.0, -30.0, 202.22000122070312],
            26: [15.0, -30.0, 202.22000122070312],
            27: [0.0, -30.0, 202.22000122070312],
            28: [-15.0, -30.0, 202.22000122070312],
            29: [-30.0, -30.0, 202.22000122070312],

            30: [30.0, -45.0, 202.22000122070312],
            31: [15.0, -45.0, 202.22000122070312],
            32: [0.0, -45.0, 202.22000122070312],
            33: [-15.0, -45.0, 202.22000122070312],
            34: [-30.0, -45.0, 202.22000122070312],

            # Правые боковые модули
            35: [67.5, 40.0, 202.22000122070312],
            36: [47.5, 40.0, 202.22000122070312],

            37: [67.5, 20.0, 202.22000122070312],
            38: [47.5, 20.0, 202.22000122070312],

            39: [67.5, 0.0, 202.22000122070312],
            40: [47.5, 0.0, 202.22000122070312],

            41: [67.5, -20.0, 202.22000122070312],
            42: [47.5, -20.0, 202.22000122070312],

            43: [67.5, -40.0, 202.22000122070312],
            44: [47.5, -40.0, 202.22000122070312],

            # Левые боковые модули
            45: [-47.5, 40.0, 202.22000122070312],
            46: [-67.5, 40.0, 202.22000122070312],

            47: [-47.5, 20.0, 202.22000122070312],
            48: [-67.5, 20.0, 202.22000122070312],

            49: [-47.5, 0.0, 202.22000122070312],
            50: [-67.5, 0.0, 202.22000122070312],

            51: [-47.5, -20.0, 202.22000122070312],
            52: [-67.5, -20.0, 202.22000122070312],

            53: [-47.5, -40.0, 202.22000122070312],
            54: [-67.5, -40.0, 202.22000122070312],
        }

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

      

        central_mask = torch.tensor([
            [0,0,1,1,1,1,1,0,0],  #  45
            [0,0,0,0,0,0,0,0,0],  #  40
            [0,0,1,1,1,1,1,0,0],  #  30
            [0,0,0,0,0,0,0,0,0],  #  20
            [0,0,1,1,1,1,1,0,0],  #  15
            [0,0,1,1,0,1,1,0,0],  #   0
            [0,0,1,1,1,1,1,0,0],  # -15
            [0,0,0,0,0,0,0,0,0],  # -20
            [0,0,1,1,1,1,1,0,0],  # -30
            [0,0,0,0,0,0,0,0,0],  # -40
            [0,0,1,1,1,1,1,0,0],  # -45
        ], dtype=torch.float32)

        side_mask = torch.tensor([
            [0,0,0,0,0,0,0,0,0],  #  45
            [1,1,0,0,0,0,0,1,1],  #  40
            [0,0,0,0,0,0,0,0,0],  #  30
            [1,1,0,0,0,0,0,1,1],  #  20
            [0,0,0,0,0,0,0,0,0],  #  15
            [1,1,0,0,0,0,0,1,1],  #   0
            [0,0,0,0,0,0,0,0,0],  # -15
            [1,1,0,0,0,0,0,1,1],  # -20
            [0,0,0,0,0,0,0,0,0],  # -30
            [1,1,0,0,0,0,0,1,1],  # -40
            [0,0,0,0,0,0,0,0,0],  # -45
        ], dtype=torch.float32)



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

        for module_id, (x, y, z) in module_coords.items():

            row = y_coords.index(y)
            col = x_coords.index(x)

            coord_X[row, col] = x
            coord_Y[row, col] = y

  

        central_square = central_mask * (15.0 * 15.0)
        side_square = side_mask * (20.0 * 20.0)

        central_depth = central_mask * 7.0
        side_depth = side_mask * 10.0

 
    

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

        self.central_fc = nn.Linear(
            1024 + self.condition_dim,
            self.hidden_channels * self.height * self.width,
        )

    
        self.central_conv1 = nn.Conv2d(
            in_channels=self.hidden_channels + 8,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.central_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

   
        self.energy_out = nn.Conv2d(
            in_channels=32,
            out_channels=17,
            kernel_size=1,
            stride=1,
            padding=0,
        )


    def _prepare_condition(self, y):

        point, momentum = y

        if self.add_points_norms_and_angles:
            point = aux.add_angle_and_norm(point)

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
        noise,
        y,
    ):

   
        condition = self._prepare_condition(y)
        x = torch.cat(
            [noise, condition],
            dim=1,
        )

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = torch.cat(
            [x, condition],
            dim=1,
        )

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = torch.cat(
            [x, condition],
            dim=1,
        )

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)

        x = torch.cat(
            [x, condition],
            dim=1,
        )

        x = self.central_fc(x)

        x = x.view(
            x.size(0),
            self.hidden_channels,
            self.height,
            self.width,
        )

#    для увеличения количества событий  в рамках 1 батча

        batch_size = x.size(0)

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
                x,
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

     

        x = self.central_conv1(x)
        x = self.activation(x)

        x = self.central_conv2(x)
        x = self.activation(x)

        density= self.energy_out(x)

    

        central_density = density[:, :7]
        side_density = density[:, 7:]

  
        central_density = (
            central_density * central_mask
        )

        side_density = (
            side_density * side_mask
        )

        physical_density = torch.zeros(
            density.size(0),
            10,
            self.height,
            self.width,
            device=density.device,
            dtype=density.dtype,
        )

        physical_density[:, :7] = (
            central_density
            + side_density[:, :7]
        )

        physical_density[:, 7:] = (
            side_density[:, 7:]
        )

        return physical_density