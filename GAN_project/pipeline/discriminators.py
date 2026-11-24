from abc import abstractmethod
from typing import Tuple, Any
from torch.nn.utils import spectral_norm
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

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


class GINEBlock(nn.Module):
    def __init__(self, hidden, edge_feature_count):
        super().__init__()
        self.conv = GINEConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            ),
            edge_dim=edge_feature_count,
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, node_features, edge_index, edge_features):
        updates = self.conv(node_features, edge_index, edge_features)
        return self.norm(node_features + updates)


class CaloganPhysicsDiscriminator3D(Discriminator):
    def __init__(self, geometry, condition_dim, hidden=64, depth=4):
        super().__init__()
        self.geometry = geometry
        self._graph_batch = None
        self.register_load_state_dict_post_hook(self._clear_graph_batch)
        energy_feature_count = 2 * geometry.max_layers + 1
        geometry_feature_count = 6
        node_input_size = energy_feature_count + geometry_feature_count + hidden

        self.condition = nn.Sequential(
            nn.Linear(condition_dim, hidden),
            nn.SiLU(),
        )
        self.input = nn.Linear(node_input_size, hidden)
        self.blocks = nn.ModuleList(
            [GINEBlock(hidden, geometry.edge_features.size(-1)) for _ in range(depth)]
        )
        self.output = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def _clear_graph_batch(self, module, incompatible_keys):
        self._graph_batch = None

    def _get_graph_batch(self, batch_size, module_count):
        edge_index = self.geometry.edge_index
        edge_features = self.geometry.edge_features
        graph_batch = self._graph_batch
        if graph_batch is not None:
            cached_edges, cached_features, cached_node_batch = graph_batch
            if (cached_node_batch.numel() == batch_size * module_count
                    and cached_edges.device == edge_index.device
                    and cached_features.dtype == edge_features.dtype):
                return graph_batch

        with torch.inference_mode(False):
            event_ids = torch.arange(batch_size, device=edge_index.device)
            offsets = event_ids * module_count
            batched_edges = (edge_index[:, None, :] + offsets[None, :, None]).flatten(1)
            batched_features = edge_features.repeat(batch_size, 1)
            node_batch = event_ids.repeat_interleave(module_count)
        graph_batch = batched_edges, batched_features, node_batch
        self._graph_batch = graph_batch
        return graph_batch

    def forward(self, energy, conditions):
        masked_energy = energy * self.geometry.layer_mask
        batch_size, module_count, _ = masked_energy.shape
        log_energy = torch.log1p(masked_energy.clamp_min(0))
        module_energy = masked_energy.sum(dim=-1, keepdim=True)

        geometry_features = self.geometry.node_features.expand(batch_size, -1, -1)
        condition_features = self.condition(conditions)
        condition_features = condition_features[:, None].expand(-1, module_count, -1)

        node_inputs = torch.cat(
            (masked_energy, log_energy, module_energy, geometry_features, condition_features),
            dim=-1,
        )
        edge_index, edge_features, node_batch = self._get_graph_batch(batch_size, module_count)
        node_features = self.input(node_inputs.flatten(0, 1))
        for block in self.blocks:
            node_features = block(node_features, edge_index, edge_features)

        mean_features = global_mean_pool(node_features, node_batch, size=batch_size)
        sum_features = global_add_pool(node_features, node_batch, size=batch_size)
        event_features = torch.cat((mean_features, sum_features), dim=-1)
        scores = self.output(event_features)
        return scores.squeeze(-1)
