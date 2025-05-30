from typing import List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import BaseNet, NetworkOutput, Action # Assuming HiddenState is also from here or torch.Tensor
# Assuming attention.py exists and AttentionEncoder is importable
from .attention import AttentionEncoder


def init(module, weight_init, bias_init, gain=1):
    """Initializes module weights and biases."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def mlp(
    input_size: int,
    layer_sizes: List[int],
    output_size: int,
    use_orthogonal: bool = True,
    use_ReLU: bool = True,
    use_value_out: bool = False,
):
    """Creates a Multi-Layer Perceptron (MLP).

    Parameters
    ----------
    input_size: int
        Dimension of the input features.
    layer_sizes: list
        List of hidden layer sizes.
    output_size: int
        Dimension of the output.
    use_orthogonal: bool
        Whether to use orthogonal initialization.
    use_ReLU: bool
        Whether to use ReLU (True) or Tanh (False) activation.
    use_value_out: bool
        Whether to apply special initialization for value/policy output layers.
    """
    active_func = nn.ReLU() if use_ReLU else nn.Tanh()
    init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
    gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')

    def init_(m):
        return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(init_(nn.Linear(sizes[i], sizes[i + 1])))
        if i < len(sizes) - 2:  # No activation or norm on the final output layer
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(active_func)

    if use_value_out and layers:
        # Find the last Linear layer and re-initialize for value/policy outputs
        last_linear_idx = -1
        for idx, layer_module in reversed(list(enumerate(layers))):
            if isinstance(layer_module, nn.Linear):
                last_linear_idx = idx
                break
        if last_linear_idx != -1:
            layers[last_linear_idx].weight.data.uniform_(-1e-3, 1e-3)
            layers[last_linear_idx].bias.data.fill_(0.0)

    return nn.Sequential(*layers)


class GraphConvLayer(nn.Module):
    """A simple graph convolutional layer."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lin_layer = nn.Linear(input_dim, output_dim)

    def forward(self, input_feature: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # input_feature: [B, N, D_in]
        # adj_matrix: [B, N, N]
        feat = self.lin_layer(input_feature)  # [B, N, D_out]
        feat = torch.bmm(adj_matrix, feat)    # [B, N, D_out] (Aggregate features)
        return feat


class GraphNetNN(nn.Module):
    """A 2-layer Graph Neural Network with skip connections and pooling."""
    def __init__(self, sa_dim: int, n_agents: int, hidden_size: int, output_dim: int,
                 agent_id: int = 0, pool_type: str = 'avg', use_agent_id: bool = False):
        """
        Args:
            sa_dim (int): Dimension of input features per agent.
            n_agents (int): Number of agents.
            hidden_size (int): Dimension of GNN hidden layers.
            output_dim (int): Dimension of the final output.
            agent_id (int, optional): Index for 'current' agent if use_agent_id is True.
            pool_type (str, optional): Pooling type ('avg' or 'max').
            use_agent_id (bool, optional): Whether to concatenate agent ID embeddings.
        """
        super().__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
        self.use_agent_id = use_agent_id
        self.output_dim = output_dim

        self.register_buffer('adj', torch.ones(n_agents, n_agents))  # Fully connected

        agent_id_attr_dim = 0
        if use_agent_id:
            agent_id_attr_dim = 2  # Example dimension for agent ID embedding
            self.agent_id = agent_id
            self.curr_agent_attr = nn.Parameter(torch.randn(agent_id_attr_dim))
            self.other_agent_attr = nn.Parameter(torch.randn(agent_id_attr_dim))
            agent_atts = [
                self.curr_agent_attr.view(1, 1, -1) if k == self.agent_id
                else self.other_agent_attr.view(1, 1, -1)
                for k in range(self.n_agents)
            ]
            self.agent_att = torch.cat(agent_atts, dim=1)  # [1, N, agent_id_attr_dim]

        current_input_dim = sa_dim + agent_id_attr_dim
        self.gc1 = GraphConvLayer(current_input_dim, hidden_size)
        self.nn_gc1 = nn.Linear(current_input_dim, hidden_size)

        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, self.output_dim)
        self.V.weight.data.uniform_(-3e-3, 3e-3)
        self.V.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N * sa_dim] (flattened per-agent features)
        batch_size = x.size(0)

        expected_flat_dim = self.n_agents * self.sa_dim
        if x.shape[1] != expected_flat_dim:
            raise ValueError(f"GraphNetNN input dim mismatch. Expected {expected_flat_dim}, got {x.shape[1]}.")

        x_reshaped = x.view(batch_size, self.n_agents, self.sa_dim)  # [B, N, sa_dim]

        if self.use_agent_id:
            agent_att_batch = self.agent_att.expand(batch_size, -1, -1)  # [B, N, agent_id_attr_dim]
            x_processed = torch.cat([x_reshaped, agent_att_batch], dim=2)  # [B, N, sa_dim + agent_id_attr_dim]
        else:
            x_processed = x_reshaped  # [B, N, sa_dim]

        adj_batch = self.adj.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]

        # Layer 1
        feat_gc1 = self.gc1(x_processed, adj_batch)  # [B, N, H]
        feat_nn1 = self.nn_gc1(x_processed)          # [B, N, H]
        feat = F.relu(feat_gc1 + feat_nn1)
        feat = F.layer_norm(feat, [feat.size(-1)])   # [B, N, H]

        # Layer 2
        out_gc2 = self.gc2(feat, adj_batch)          # [B, N, H]
        out_nn2 = self.nn_gc2(feat)                  # [B, N, H]
        out = F.relu(out_gc2 + out_nn2)
        out = F.layer_norm(out, [out.size(-1)])      # [B, N, H]

        # Pool
        if self.pool_type == 'avg':
            ret = out.mean(dim=1)  # [B, H]
        elif self.pool_type == 'max':
            ret, _ = out.max(dim=1) # [B, H]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        output_val = self.V(ret)  # [B, output_dim]
        return output_val


class RepresentationNetwork(nn.Module):
    """Encodes observations into hidden states."""
    def __init__(
        self,
        observation_size: int,
        hidden_state_size: int,
        fc_representation_layers: List[int],
        use_feature_norm: bool = True,
    ):
        super().__init__()
        self.use_feature_norm = use_feature_norm
        self.feature_norm = nn.LayerNorm(observation_size) if use_feature_norm else nn.Identity()
        self.mlp = mlp(observation_size, fc_representation_layers, hidden_state_size, use_ReLU=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, Obs_Size] (flattened observations per agent)
        x = self.feature_norm(x)
        x = self.mlp(x)  # [B*N, H] (hidden state per agent)
        return x


class DynamicsNetwork(nn.Module):
    """Predicts next hidden state and reward."""
    def __init__(
        self,
        num_agents: int,
        hidden_state_size: int, # H (per agent)
        action_space_size: int, # A (per agent)
        fc_dynamic_layers: List[int],
        fc_reward_layers: List[int], # Used for MLP reward head
        reward_support_size: int,
        reward_head_type: str = 'gnn', # 'mlp' or 'gnn'
        gnn_reward_hidden_size: int = 64, # Hidden size for GNN reward head
    ):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_state_size = hidden_state_size
        self.action_space_size = action_space_size
        self.reward_head_type = reward_head_type

        # --- Next State Prediction Components ---
        # Input to attention is per-agent: hidden_state + action_onehot
        attn_input_dim = hidden_state_size + action_space_size
        self.attention_stack = nn.Sequential(
            nn.Linear(attn_input_dim, hidden_state_size),
            nn.ReLU(),
            AttentionEncoder(3, hidden_state_size, hidden_state_size, dropout=0.1) # Assuming 3 heads
        )
        # Input to dynamics MLP is per-agent: hidden_state + action_onehot + attention_output
        dynamic_mlp_input_size = hidden_state_size + action_space_size + hidden_state_size
        self.fc_dynamic = mlp(dynamic_mlp_input_size, fc_dynamic_layers, hidden_state_size)

        # --- Reward Prediction Head ---
        reward_feature_dim_per_agent = hidden_state_size + action_space_size # Features: next_hs + action
        if self.reward_head_type == 'gnn':
            self.reward_predictor = GraphNetNN(
                sa_dim=reward_feature_dim_per_agent,
                n_agents=num_agents,
                hidden_size=gnn_reward_hidden_size,
                output_dim=reward_support_size,
                pool_type='avg',
                use_agent_id=False
            )
        elif self.reward_head_type == 'mlp':
            mlp_reward_input_dim = num_agents * reward_feature_dim_per_agent
            self.reward_predictor = mlp(
                mlp_reward_input_dim,
                fc_reward_layers,
                reward_support_size,
                use_value_out=True # Typical for reward/value heads
            )
        else:
            raise ValueError(f"Unknown reward_head_type: {reward_head_type}")

    def forward(self, hidden_state: torch.Tensor, action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_state: [B, N, H]
        # action_onehot: [B, N, A]
        batch_size, num_agents, _ = hidden_state.shape
        pre_state = hidden_state  # For residual connection

        # --- Predict Next State ---
        attn_input = torch.cat([hidden_state, action_onehot], dim=2)  # [B, N, H+A]
        attn_output = self.attention_stack(attn_input)               # [B, N, H] (assuming AttentionEncoder output)

        # Each agent's dynamics input: current_hs, action, attention_output
        concat_input_dynamic = torch.cat([hidden_state, action_onehot, attn_output], dim=2)  # [B, N, H+A+H]
        # Process each agent independently for state update
        concat_input_dynamic_flat = concat_input_dynamic.reshape(batch_size * num_agents, -1) # [B*N, H+A+H]
        state_update = self.fc_dynamic(concat_input_dynamic_flat)    # [B*N, H]
        state_update = state_update.reshape(batch_size, num_agents, self.hidden_state_size) # [B, N, H]
        next_hidden_state = state_update + pre_state                # [B, N, H] (Residual connection)

        # --- Predict Reward ---
        # Features for reward: predicted_next_state + action_taken
        reward_features_per_agent = torch.cat([next_hidden_state, action_onehot], dim=2) # [B, N, H+A]

        if self.reward_head_type == 'gnn':
            # GNN expects [B, N*sa_dim]
            reward_input_flat = reward_features_per_agent.reshape(batch_size, -1) # [B, N*(H+A)]
            reward_logits = self.reward_predictor(reward_input_flat)              # [B, reward_support_size]
        elif self.reward_head_type == 'mlp':
            # MLP expects [B, N*(H+A)]
            reward_input_flat = reward_features_per_agent.reshape(batch_size, -1) # [B, N*(H+A)]
            reward_logits = self.reward_predictor(reward_input_flat)              # [B, reward_support_size]

        return next_hidden_state, reward_logits


class PredictionNetwork(nn.Module):
    """Predicts policy and value from hidden state."""
    def __init__(
        self,
        num_agents: int,
        hidden_state_size: int, # H (per agent)
        action_space_size: int, # A (per agent)
        fc_value_layers: List[int], # Used for MLP value head
        fc_policy_layers: List[int],
        value_support_size: int,
        value_head_type: str = 'gnn', # 'mlp' or 'gnn'
        gnn_value_hidden_size: int = 64, # Hidden size for GNN value head
    ):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_state_size_per_agent = hidden_state_size # For clarity
        self.action_space_size = action_space_size
        self.value_head_type = value_head_type

        # --- Value Prediction Head ---
        # Input feature per agent for GNN value prediction: hidden_state
        value_feature_dim_per_agent = self.hidden_state_size_per_agent
        if self.value_head_type == 'gnn':
            self.value_predictor = GraphNetNN(
                sa_dim=value_feature_dim_per_agent,
                n_agents=num_agents,
                hidden_size=gnn_value_hidden_size,
                output_dim=value_support_size,
                pool_type='avg',
                use_agent_id=False
            )
        elif self.value_head_type == 'mlp':
            mlp_value_input_dim = num_agents * value_feature_dim_per_agent
            self.value_predictor = mlp(
                mlp_value_input_dim,
                fc_value_layers,
                value_support_size,
                use_value_out=True
            )
        else:
            raise ValueError(f"Unknown value_head_type: {value_head_type}")

        # --- Policy Prediction Head (always MLP per agent) ---
        self.fc_policy = mlp(
            self.hidden_state_size_per_agent,
            fc_policy_layers,
            action_space_size,
            use_value_out=True # Common for policy heads
        )

    def forward(self, hidden_state_total: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_state_total: [B, N*H] (flattened hidden state across all agents)
        batch_size = hidden_state_total.size(0)

        expected_flat_dim = self.num_agents * self.hidden_state_size_per_agent
        if hidden_state_total.shape[1] != expected_flat_dim:
            # Try to infer if input was [B, N, H]
            if hidden_state_total.dim() == 3 and \
               hidden_state_total.shape[1] == self.num_agents and \
               hidden_state_total.shape[2] == self.hidden_state_size_per_agent:
                hidden_state_per_agent = hidden_state_total # [B, N, H]
                hidden_state_total_for_value = hidden_state_per_agent.reshape(batch_size, -1) # [B, N*H]
            else:
                raise ValueError(f"PredictionNetwork input dim mismatch. Expected flat {expected_flat_dim} "
                                 f"or [B, {self.num_agents}, {self.hidden_state_size_per_agent}], "
                                 f"got {hidden_state_total.shape}")
        else: # Input is already [B, N*H]
            hidden_state_per_agent = hidden_state_total.view(
                batch_size, self.num_agents, self.hidden_state_size_per_agent
            ) # [B, N, H]
            hidden_state_total_for_value = hidden_state_total # [B, N*H]


        # --- Value Prediction ---
        # Both GNN and MLP value heads expect the flattened total hidden state [B, N*H]
        # GNN internally reshapes it to [B, N, H] if its sa_dim is H per agent.
        value_logits = self.value_predictor(hidden_state_total_for_value)  # [B, value_support_size]

        # --- Policy Prediction (per agent) ---
        # Reshape hidden state to process each agent independently for policy
        policy_input_flat = hidden_state_per_agent.reshape(
            batch_size * self.num_agents, self.hidden_state_size_per_agent
        ) # [B*N, H]
        policy_logits_flat = self.fc_policy(policy_input_flat)  # [B*N, A]
        policy_logits = policy_logits_flat.reshape(
            batch_size, self.num_agents, self.action_space_size
        ) # [B, N, A]

        return policy_logits, value_logits


class ProjectionNetwork(nn.Module):
    """Projection and Prediction heads for self-supervised learning (e.g., SimSiam-like)."""
    def __init__(
        self,
        projection_in_dim: int, # Expected: N * H (total hidden state dim)
        proj_hid: int = 256,
        proj_out: int = 256,
        pred_hid: int = 64,
        pred_out: int = 256,
    ):
        super().__init__()
        self.projection_in_dim = projection_in_dim
        self.projection = mlp(projection_in_dim, [proj_hid], proj_out, use_ReLU=True)
        self.projection_norm = nn.LayerNorm(proj_out) # Norm after projection
        self.prediction = mlp(proj_out, [pred_hid], pred_out, use_ReLU=True)

    def project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state: [B, N*H]
        if hidden_state.shape[1] != self.projection_in_dim:
             raise ValueError(f"ProjectionNetwork.project input dim. Expected {self.projection_in_dim}, got {hidden_state.shape[1]}")
        proj = self.projection(hidden_state) # [B, proj_out]
        proj = self.projection_norm(proj)    # [B, proj_out]
        return proj

    def predict(self, projection: torch.Tensor) -> torch.Tensor:
        # projection: [B, proj_out]
        return self.prediction(projection) # [B, pred_out]


class MAMuZeroNet(BaseNet):
    """Multi-Agent MuZero Network."""
    def __init__(
        self,
        num_agents: int,
        observation_shape: Tuple[int, ...], # e.g., (C, W, H) or (Features,) per agent
        action_space_size: int, # Per agent
        hidden_state_size: int, # Per agent hidden state size (H)
        fc_representation_layers: List[int],
        fc_dynamic_layers: List[int],
        fc_reward_layers: List[int],    # Used if reward_head_type is 'mlp'
        fc_value_layers: List[int],     # Used if value_head_type is 'mlp'
        fc_policy_layers: List[int],
        reward_support_size: int,
        value_support_size: int,
        inverse_value_transform: Any,
        inverse_reward_transform: Any,
        reward_head_type: str = 'gnn',   # 'mlp' or 'gnn'
        value_head_type: str = 'gnn',    # 'mlp' or 'gnn'
        gnn_reward_hidden_size: int = 64,# Hidden size for GNN reward head
        gnn_value_hidden_size: int = 64, # Hidden size for GNN value head
        proj_hid: int = 256,
        proj_out: int = 256,
        pred_hid: int = 64,
        pred_out: int = 256,
        use_feature_norm: bool = True,
        **kwargs # Absorb any other legacy arguments
    ):
        super().__init__(inverse_value_transform, inverse_reward_transform)
        self.num_agents = num_agents
        self.obs_size_per_agent = np.prod(observation_shape)
        self.action_space_size = action_space_size
        self.hidden_state_size_per_agent = hidden_state_size
        self.total_hidden_state_size = num_agents * hidden_state_size # N*H

        self.representation_network = RepresentationNetwork(
            self.obs_size_per_agent,
            self.hidden_state_size_per_agent, # Outputs H per agent
            fc_representation_layers,
            use_feature_norm
        )

        self.dynamics_network = DynamicsNetwork(
            num_agents,
            self.hidden_state_size_per_agent,
            action_space_size,
            fc_dynamic_layers,
            fc_reward_layers, # For MLP case
            reward_support_size,
            reward_head_type=reward_head_type,
            gnn_reward_hidden_size=gnn_reward_hidden_size # For GNN case
        )

        self.prediction_network = PredictionNetwork(
            num_agents,
            self.hidden_state_size_per_agent,
            action_space_size,
            fc_value_layers, # For MLP case
            fc_policy_layers,
            value_support_size,
            value_head_type=value_head_type,
            gnn_value_hidden_size=gnn_value_hidden_size # For GNN case
        )

        self.projection_network = ProjectionNetwork(
            self.total_hidden_state_size, # Projection input is [B, N*H]
            proj_hid, proj_out, pred_hid, pred_out,
        )

    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        # observation: [B, N, ObsFeatures] or [B, N, C, W, H]
        batch_size = observation.shape[0]

        # Flatten observation to [B*N, Obs_Size_Per_Agent]
        obs_flat_per_agent = observation.reshape(batch_size * self.num_agents, -1)
        if obs_flat_per_agent.shape[1] != self.obs_size_per_agent:
            raise ValueError(f"Observation feature size mismatch. Expected {self.obs_size_per_agent}, "
                             f"got {obs_flat_per_agent.shape[1]} from input {observation.shape}")

        # Repr net outputs [B*N, H_per_agent]
        hidden_state_flat_per_agent = self.representation_network(obs_flat_per_agent)
        # Reshape to [B, N, H_per_agent]
        hidden_state_per_agent = hidden_state_flat_per_agent.reshape(
            batch_size, self.num_agents, self.hidden_state_size_per_agent
        )
        # Return concatenated hidden state [B, N*H]
        return hidden_state_per_agent.reshape(batch_size, -1)


    def prediction(self, hidden_state_total: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_state_total: [B, N*H]
        if hidden_state_total.shape[1] != self.total_hidden_state_size:
             raise ValueError(f"Prediction input dim. Expected {self.total_hidden_state_size}, got {hidden_state_total.shape[1]}")
        policy_logits, value_logits = self.prediction_network(hidden_state_total)
        # policy_logits: [B, N, A]
        # value_logits: [B, value_support_size]
        return policy_logits, value_logits

    def dynamics(self, hidden_state_total: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_state_total: [B, N*H]
        # action: [B, N] (action indices per agent)
        batch_size = hidden_state_total.shape[0]

        # Reshape hidden_state to [B, N, H] for dynamics network
        hidden_state_per_agent = hidden_state_total.view(
            batch_size, self.num_agents, self.hidden_state_size_per_agent
        ) # [B, N, H]

        # Convert action indices to one-hot vectors [B, N, A]
        action_onehot = F.one_hot(action.long(), num_classes=self.action_space_size).float()
        # Ensure shape [B, N, A] if action came as [B*N]
        if action_onehot.dim() == 2 and action_onehot.shape[0] == batch_size * self.num_agents:
             action_onehot = action_onehot.view(batch_size, self.num_agents, self.action_space_size)
        elif not (action_onehot.dim() == 3 and action_onehot.shape[0] == batch_size and \
                  action_onehot.shape[1] == self.num_agents and action_onehot.shape[2] == self.action_space_size):
             raise ValueError(f"Action one-hot shape error. Got {action_onehot.shape} from action {action.shape}")

        # Dynamics net: ([B,N,H], [B,N,A]) -> ([B,N,H], [B, reward_support])
        next_hidden_state_per_agent, reward_logits = self.dynamics_network(hidden_state_per_agent, action_onehot)

        # Reshape next_hidden_state back to [B, N*H]
        next_hidden_state_total = next_hidden_state_per_agent.reshape(batch_size, -1)
        return next_hidden_state_total, reward_logits


    def project(self, hidden_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        # hidden_state: [B, N*H]
        if hidden_state.shape[1] != self.total_hidden_state_size:
            raise ValueError(f"Project input dim. Expected {self.total_hidden_state_size}, got {hidden_state.shape[1]}")

        proj = self.projection_network.project(hidden_state) # [B, proj_out]
        if with_grad:
            return self.projection_network.predict(proj) # [B, pred_out]
        else:
            return proj.detach()


    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        # observation: [B, N, ObsFeatures] or [B, N, C, W, H]
        batch_size = observation.size(0)
        hidden_state = self.representation(observation) # [B, N*H]
        policy_logits, value_logits = self.prediction(hidden_state) # [B,N,A], [B, value_support]

        reward_logits = torch.zeros(batch_size, self.dynamics_network.reward_predictor.output_dim, device=observation.device) # [B, reward_support]
        # ^ Assuming reward_predictor.output_dim gives reward_support_size for both GNN/MLP

        if not self.training:
            value_scalar = self.inverse_value_transform(value_logits).detach().cpu().numpy()
            # For initial step, reward is typically 0 or not predicted from observation alone.
            # Using inverse_reward_transform on zeros will likely give zeros.
            reward_scalar = self.inverse_reward_transform(reward_logits).detach().cpu().numpy()
            policy_logits_np = policy_logits.detach().cpu().numpy()
            return NetworkOutput(hidden_state=hidden_state, reward=reward_scalar, value=value_scalar, policy_logits=policy_logits_np)
        else:
            return NetworkOutput(hidden_state=hidden_state, reward=reward_logits, value=value_logits, policy_logits=policy_logits)


    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        # hidden_state: [B, N*H]
        # action: [B, N] (action indices)
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action) # [B,N*H], [B, reward_support]
        policy_logits, value_logits = self.prediction(next_hidden_state)      # [B,N,A], [B, value_support]

        if not self.training:
            reward_scalar = self.inverse_reward_transform(reward_logits).detach().cpu().numpy()
            value_scalar = self.inverse_value_transform(value_logits).detach().cpu().numpy()
            policy_logits_np = policy_logits.detach().cpu().numpy()
            return NetworkOutput(hidden_state=next_hidden_state, reward=reward_scalar, value=value_scalar, policy_logits=policy_logits_np)
        else:
            return NetworkOutput(hidden_state=next_hidden_state, reward=reward_logits, value=value_logits, policy_logits=policy_logits)