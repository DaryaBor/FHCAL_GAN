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
    """Fixed-resolution XY CNN with dilated blocks and physical integration.

    Every spatial tensor uses geometry.height x geometry.width from the start.
    The historical class name is retained; layers are Conv2d, depth is channels.
    """
    def __init__(self, geometry, condition_dim, noise_dim=64, hidden=64,
                 condition_state=None, dilation_rates=(1, 2, 4), *, energy_condition=False):
        super().__init__()
        if len(dilation_rates) != 3 or any(not isinstance(d, int) or d < 1 for d in dilation_rates):
            raise ValueError('dilation_rates must contain three positive integers')
        self.geometry, self.noise_dim = geometry, noise_dim
        # Persist the dataset encoder for inference in the existing GAN checkpoint.
        state = condition_state or {'mean': torch.zeros(6), 'scale': torch.ones(6),
                                    'pdg_codes': list(range(condition_dim-6))}
        self.register_buffer('condition_mean', state['mean'].clone())
        self.register_buffer('condition_scale', state['scale'].clone())
        self.register_buffer('pdg_codes', torch.tensor(state['pdg_codes'], dtype=torch.long))

        self.condition = nn.Sequential(
            nn.Linear(condition_dim + int(energy_condition), hidden),
            nn.SiLU(),
        )
        # Shared event features + six spatial geometry channels at full H x W.
        self.input = nn.Sequential(
            nn.Conv2d(noise_dim + hidden + 6, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Each block combines expanded spatial coverage with local processing.
        # padding=dilation preserves H x W for a 3x3 kernel, stride=1.
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3,
                      dilation=dilation_rates[0], padding=dilation_rates[0]),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3,
                      dilation=dilation_rates[1], padding=dilation_rates[1]),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3,
                      dilation=dilation_rates[2], padding=dilation_rates[2]),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.output = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, geometry.max_layers, kernel_size=1),
            nn.Softplus(),
        )

    def virtual_density(self, noise, conditions):
        # The same global noise/conditions are available at every physical position.
        event = torch.cat((noise, self.condition(conditions)), dim=1)
        event_map = event[:, :, None, None].expand(
            -1, -1, self.geometry.height, self.geometry.width,
        )
        static = self.geometry.grid_features.expand(noise.size(0), -1, -1, -1)
        h = self.input(torch.cat((event_map, static), dim=1))
        h = F.silu(h + self.block1(h))
        h = F.silu(h + self.block2(h))
        h = F.silu(h + self.block3(h))
        return self.output(h) * self.geometry.virtual_mask

    def forward(self, noise, conditions):
        return self.geometry.aggregate(self.virtual_density(noise, conditions))


class ConditionalEnergyFlow(nn.Module):
    def __init__(self, condition_dim, hidden=64):
        super().__init__()
        self.velocity = nn.Sequential(
            nn.Linear(condition_dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.positive_logit = nn.Sequential(
            nn.Linear(condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer('log_mean', torch.zeros(()))
        self.register_buffer('log_scale', torch.ones(()))
        self.register_buffer('has_empty_events', torch.tensor(False))

    def standardize(self, energy):
        return (energy.log() - self.log_mean) / self.log_scale

    def _velocity(self, position, time, conditions):
        inputs = torch.cat((conditions, position[:, None], time[:, None]), dim=1)
        return self.velocity(inputs).squeeze(1)

    def fit(self, energy, conditions, *, updates=3000, batch_size=256):
        device = self.log_mean.device
        energy = energy.to(device)
        conditions = conditions.to(device)
        positive = energy > 0
        if not positive.any():
            raise ValueError('Energy flow requires positive total energies in the training data.')

        positive_logs = energy[positive].log()
        positive_conditions = conditions[positive]
        self.log_mean.copy_(positive_logs.mean())
        self.log_scale.copy_(positive_logs.std(unbiased=False).clamp_min(1e-6))
        self.has_empty_events.copy_((~positive).any())
        targets = (positive_logs - self.log_mean) / self.log_scale
        fit_empty_events = bool(self.has_empty_events.item())

        self.requires_grad_(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        rng = torch.Generator(device=device).manual_seed(2026)
        for _ in range(updates):
            indices = torch.randint(len(targets), (batch_size,), device=device, generator=rng)
            target = targets[indices]
            noise = torch.randn(target.shape, device=device, generator=rng)
            time = torch.rand(target.shape, device=device, generator=rng)
            position = torch.lerp(noise, target, time)
            velocity = self._velocity(position, time, positive_conditions[indices])
            loss = F.mse_loss(velocity, target - noise)
            if fit_empty_events:
                indices = torch.randint(len(energy), (batch_size,), device=device, generator=rng)
                logits = self.positive_logit(conditions[indices]).squeeze(1)
                loss = loss + F.binary_cross_entropy_with_logits(logits, positive[indices].to(logits.dtype))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        self.requires_grad_(False)
        if not all(torch.isfinite(parameter).all() for parameter in self.parameters()):
            raise FloatingPointError('Energy-flow fitting produced nonfinite parameters.')

    @torch.no_grad()
    def sample(self, conditions):
        position = torch.randn_like(conditions[:, 0])
        dt = 1.0 / 32
        for step in range(32):
            time = torch.full_like(position, step * dt)
            midpoint = position + dt / 2 * self._velocity(position, time, conditions)
            position = position + dt * self._velocity(midpoint, time + dt / 2, conditions)

        energy = (self.log_mean + self.log_scale * position).exp()
        if not (torch.isfinite(energy) & (energy > 0)).all():
            raise FloatingPointError('Sampled positive energies are outside the representable range.')
        if bool(self.has_empty_events.item()):
            positive_probability = self.positive_logit(conditions).squeeze(1).sigmoid()
            energy = energy * (torch.rand_like(energy) < positive_probability)
        return energy


class FlowFractionGenerator(CaloganPhysicsGenerator3D):
    def __init__(self, geometry, condition_dim, noise_dim=64, hidden=64,
                 condition_state=None, dilation_rates=(1, 2, 4)):
        super().__init__(
            geometry, condition_dim, noise_dim=noise_dim, hidden=hidden,
            condition_state=condition_state, dilation_rates=dilation_rates, energy_condition=True,
        )
        self.energy_flow = ConditionalEnergyFlow(condition_dim)
        self.energy_flow.requires_grad_(False)

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.energy_flow.requires_grad_(False)
        return self

    def energy_fractions(self, noise, conditions, total_energy):
        positive = total_energy > 0
        safe_energy = torch.where(positive, total_energy, torch.ones_like(total_energy))
        energy_condition = self.energy_flow.standardize(safe_energy)
        energy_condition = torch.where(positive, energy_condition, torch.zeros_like(energy_condition))
        conditions = torch.cat((conditions, energy_condition[:, None]), dim=1)
        weights = super().forward(noise, conditions)
        weights = weights + self.geometry.layer_mask * torch.finfo(weights.dtype).tiny
        return weights / weights.sum(dim=(1, 2), keepdim=True)

    def forward(self, noise, conditions, *, total_energy=None):
        if total_energy is None:
            total_energy = self.energy_flow.sample(conditions)
        fractions = self.energy_fractions(noise, conditions, total_energy)
        return total_energy[:, None, None] * fractions
