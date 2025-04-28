# --- Imports remain the same ---
from typing import List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import BaseNet, NetworkOutput, Action, HiddenState
# Assuming attention.py exists in the same directory or is importable
from .attention import AttentionEncoder


# --- init, mlp, GraphConvLayer, GraphNetNN remain the same as in the provided code ---

def init(module, weight_init, bias_init, gain=1):
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
    """MLP layers

    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    """

    active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
    gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

    def init_(m):
        return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        # Add LayerNorm before activation, consistent with typical modern practice
        layers.append(init_(nn.Linear(sizes[i], sizes[i + 1])))
        if i < len(sizes) - 2: # No activation or norm on the final output layer
             layers.append(nn.LayerNorm(sizes[i + 1]))
             layers.append(active_func)

    if use_value_out:
        # Re-initialize the last linear layer specifically for value/policy outputs
        # Find the last Linear layer
        last_linear_idx = -1
        for idx, layer in reversed(list(enumerate(layers))):
            if isinstance(layer, nn.Linear):
                last_linear_idx = idx
                break
        if last_linear_idx != -1:
             # Special init for value head: small weights, zero bias
             layers[last_linear_idx].weight.data.uniform_(-1e-3, 1e-3) # Or fill_(0)
             layers[last_linear_idx].bias.data.fill_(0)


    return nn.Sequential(*layers)

class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin_layer = nn.Linear(input_dim, output_dim)  # Process per-agent features

    def forward(self, input_feature, adj_matrix):
        # input_feature: [B, N, D] (batch, num_agents, feature_dim)
        # adj_matrix: [B, N, N]
        feat = self.lin_layer(input_feature)  # [B, N, H]
        feat = torch.bmm(adj_matrix, feat)     # [B, N, H]
        return feat

class GraphNetNN(nn.Module):
    """
    A permutation-invariant graph net. Can be used for value function approximation
    (outputting logits for a value distribution) or other tasks requiring
    a pooled representation mapped to a specific output dimension.
    Takes a flattened state input and processes it as a graph.
    """

    def __init__(self, sa_dim, n_agents, hidden_size, output_dim, # <-- Added output_dim
                 agent_id=0, pool_type='avg', use_agent_id=False):
        """
        Args:
            sa_dim (int): Dimension of the input features per agent (after potential reshaping).
            n_agents (int): Number of agents.
            hidden_size (int): Dimension of the hidden layers in the GNN.
            output_dim (int): Dimension of the final output layer (e.g., 1 for scalar value,
                              value_support_size for value distribution logits).
            agent_id (int, optional): Index of the 'current' agent if use_agent_id is True. Defaults to 0.
            pool_type (str, optional): Pooling type ('avg' or 'max'). Defaults to 'avg'.
            use_agent_id (bool, optional): Whether to concatenate agent ID embeddings. Defaults to False.
        """
        super(GraphNetNN, self).__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
        self.use_agent_id = use_agent_id
        self.output_dim = output_dim # Store output dimension

        # Create a fully connected adjacency matrix (including self-loops)
        self.register_buffer('adj', torch.ones(n_agents, n_agents)) # Keep original behaviour for now

        agent_id_attr_dim = 0 # Default if not using agent id
        if use_agent_id:
            # Ensure agent_id_attr_dim is defined if use_agent_id is True
            agent_id_attr_dim = 2 # Or get this from config if needed
            self.agent_id = agent_id # Store agent_id only if used

            self.curr_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)
            self.other_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)

            agent_att_list = [] # Use a list for clarity
            for k in range(self.n_agents):
                if k == self.agent_id:
                    # Add batch and agent dimensions before concatenating feature dim
                    agent_att_list.append(self.curr_agent_attr.view(1, 1, agent_id_attr_dim))
                else:
                    agent_att_list.append(self.other_agent_attr.view(1, 1, agent_id_attr_dim))
            # Concatenate along the agent dimension (dim=1)
            self.agent_att = torch.cat(agent_att_list, dim=1) # Shape [1, N, agent_id_attr_dim]


        current_input_dim = sa_dim + agent_id_attr_dim
        self.gc1 = GraphConvLayer(current_input_dim, hidden_size)
        self.nn_gc1 = nn.Linear(current_input_dim, hidden_size)


        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        # The final layer maps the pooled representation to output_dim
        self.V = nn.Linear(hidden_size, self.output_dim)
        # Consider different initialization if outputting distribution logits
        # E.g., small weights, zero bias might be better than mul_(0.1)
        self.V.weight.data.uniform_(-3e-3, 3e-3) # Example init
        self.V.bias.data.fill_(0.0)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, expected shape [B, N * D],
                              where D = sa_dim. It will be reshaped internally to [B, N, D].

        Returns:
            torch.Tensor: Output tensor of shape [B, output_dim].
        """
        batch_size = x.size(0)

        # Reshape flattened input to [B, N, D]
        expected_flat_dim = self.n_agents * self.sa_dim
        if x.shape[1] != expected_flat_dim:
             raise ValueError(f"GraphNetNN input has wrong flat dimension. "
                              f"Expected {expected_flat_dim}, got {x.shape[1]}. "
                              f"Input shape: {x.shape}, n_agents: {self.n_agents}, sa_dim: {self.sa_dim}")

        try:
            # Reshape based on self.sa_dim (feature dim per agent *before* agent ID)
            x_reshaped = x.view(batch_size, self.n_agents, self.sa_dim) # [B, N, D]
        except RuntimeError as e:
            print(f"Error reshaping input in GraphNetNN. Input shape: {x.shape}, "
                  f"Target shape: ({batch_size}, {self.n_agents}, {self.sa_dim})")
            raise e

        # Add agent ID attributes if enabled
        if self.use_agent_id:
            # Expand agent attributes to batch size
            agent_att_batch = self.agent_att.expand(batch_size, -1, -1) # [B, N, agent_id_attr_dim]
            # Concatenate along the feature dimension (dim=2)
            x_processed = torch.cat([x_reshaped, agent_att_batch], dim=2) # [B, N, D + agent_id_attr_dim]
        else:
            x_processed = x_reshaped # [B, N, D]

        # Expand adjacency matrix for batch processing
        adj = self.adj.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]

        # Process through graph layers
        # Layer 1
        feat_gc1 = self.gc1(x_processed, adj) # [B, N, H]
        feat_nn1 = self.nn_gc1(x_processed)   # [B, N, H]
        feat = F.relu(feat_gc1 + feat_nn1)    # Apply ReLU after summing GCN and MLP paths
        # Normalization: LayerNorm is often more stable than dividing by N
        feat = F.layer_norm(feat, [feat.size(-1)]) # LayerNorm

        # Layer 2
        out_gc2 = self.gc2(feat, adj)         # [B, N, H]
        out_nn2 = self.nn_gc2(feat)           # [B, N, H]
        out = F.relu(out_gc2 + out_nn2)       # Apply ReLU after summing
        # Normalization
        out = F.layer_norm(out, [out.size(-1)]) # LayerNorm

        # Pool over agents
        if self.pool_type == 'avg':
            ret = out.mean(dim=1)  # [B, H]
        elif self.pool_type == 'max':
            ret, _ = out.max(dim=1) # [B, H]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Final linear layer to produce output
        output_val = self.V(ret)  # [B, output_dim]

        return output_val


# --- RepresentationNetwork remains the same ---
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        hidden_state_size: int,
        fc_representation_layers: List[int],
        use_feature_norm: bool = True,
    ):
        super().__init__()
        self.use_feature_norm = use_feature_norm
        if self.use_feature_norm:
            # Apply LayerNorm *before* the MLP
            self.feature_norm = nn.LayerNorm(observation_size)
        else:
            self.feature_norm = nn.Identity() # Placeholder if not used

        # Note: Changed mlp to use the potentially updated implementation
        self.mlp = mlp(observation_size, fc_representation_layers, hidden_state_size, use_ReLU=True) # Assuming ReLU is desired here

    def forward(self, x):
        x = self.feature_norm(x)
        x = self.mlp(x)
        return x


# --- DynamicsNetwork is MODIFIED ---
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        num_agents: int,
        hidden_state_size: int,
        action_space_size: int,
        fc_dynamic_layers: List[int],
        fc_reward_layers: List[int], # Note: fc_reward_layers now defines hidden layers *within* GraphNetNN
        full_support_size: int, # This is reward_support_size
        gnn_reward_hidden_size: int = None, # Optional: specific hidden size for reward GNN
    ):
        """Dynamics network: Predict next hidden states given current states and actions

        Parameters
        ----------
        num_agents: int
            number of agents
        hidden_state_size: int
            dim of hidden states
        action_space_size: int
            action space size per agent
        fc_dynamic_layers: list
            hidden layers of the state transition (state+action -> state) MLP
        fc_reward_layers: list
            hidden layers *inside* the reward GraphNetNN (used for its internal MLP/GCN layers)
        full_support_size: int
            dim of reward output (reward support size)
        gnn_reward_hidden_size: int, optional
            Hidden dimension size for the reward GNN layers. Defaults to hidden_state_size.
        """
        super().__init__()
        self.num_agents = num_agents
        self.hidden_state_size = hidden_state_size
        self.action_space_size = action_space_size

        # --- Dynamics part (predicting next state) ---
        self.attention_stack = nn.Sequential(
            # Input is per-agent: hidden_state + action_onehot
            nn.Linear(hidden_state_size + action_space_size, hidden_state_size),
            nn.ReLU(),
            # Assuming AttentionEncoder takes [B, N, D] or similar
            AttentionEncoder(3, hidden_state_size, hidden_state_size, dropout=0.1)
        )

        # Input to MLP is per-agent: hidden_state + action_onehot + attention_output
        dynamic_mlp_input_size = hidden_state_size + action_space_size + hidden_state_size
        self.fc_dynamic = mlp(dynamic_mlp_input_size,
                              fc_dynamic_layers, hidden_state_size)

        # --- Reward prediction part (MODIFIED) ---
        # Use GraphNetNN for permutation-invariant reward prediction
        # Input feature per agent for reward prediction: next_hidden_state + action_onehot
        reward_gnn_input_dim = hidden_state_size + action_space_size
        # Use specified GNN hidden size or default to main hidden_state_size
        reward_gnn_hidden = gnn_reward_hidden_size if gnn_reward_hidden_size is not None else hidden_state_size

        # We use GraphNetNN here. It will handle the permutation invariance.
        # fc_reward_layers are not directly used here, as GraphNetNN has its own structure.
        # However, you *could* potentially pass layer sizes *into* GraphNetNN if you modified it,
        # but the current GraphNetNN uses a fixed 2-layer GCN structure with a single `hidden_size`.
        # We'll use `reward_gnn_hidden` for that internal hidden size.
        self.reward_predictor = GraphNetNN(
            sa_dim=reward_gnn_input_dim, # state + action dim per agent
            n_agents=num_agents,
            hidden_size=reward_gnn_hidden, # Internal hidden dim for GNN
            output_dim=full_support_size,  # reward support size
            pool_type='avg', 
            use_agent_id=False 
        )
        # Note: The previous MLP fc_reward is removed.


    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor):
        # hidden_state: [B, N, H]
        # action: [B, N, A] (already one-hot encoded by caller)
        batch_size, num_agents, _ = hidden_state.shape
        pre_state = hidden_state # Store for residual connection

        # --- Predict Next State ---
        attn_input = torch.cat([hidden_state, action], dim=2) # [B, N, H+A]
        attn = self.attention_stack(attn_input) # [B, N, H] - Assuming AttentionEncoder handles [B, N, D]

        concat_input = torch.cat([hidden_state, action, attn], dim=2) # [B, N, H+A+H]
        # Reshape for MLP: process each agent independently
        concat_input_flat = concat_input.reshape(batch_size * num_agents, -1)
        # MLP predicts state update per agent
        state_update = self.fc_dynamic(concat_input_flat).reshape(batch_size, num_agents, self.hidden_state_size) # [B, N, H]
        # Residual connection for next state
        next_hidden_state = state_update + pre_state # [B, N, H]

        # --- Predict Reward (using GraphNetNN) ---
        # Input for reward GNN: uses the *predicted next state* and the action taken
        reward_gnn_features_per_agent = torch.cat([next_hidden_state, action], dim=2) # [B, N, H+A]

        # GraphNetNN expects a flattened input [B, N*D]
        reward_gnn_input_flat = reward_gnn_features_per_agent.reshape(batch_size, -1) # [B, N*(H+A)]

        # Predict reward logits using the permutation-invariant GNN
        reward_logits = self.reward_predictor(reward_gnn_input_flat) # [B, reward_support_size]

        # Return predicted next state (per agent) and predicted reward logits (collective)
        return next_hidden_state, reward_logits


# --- PredictionNetwork remains the same ---
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        num_agents: int,
        hidden_state_size: int,
        action_space_size: int,
        fc_value_layers: List[int], # Used internally by GraphNetNN (as hidden_size)
        fc_policy_layers: List[int],
        full_support_size: int, # value_support_size
        gnn_value_hidden_size: int = None, # Optional: specific hidden size for value GNN
    ):
        """Prediction network: predict the value and policy given hidden states

        Parameters
        ----------
        num_agents: int
            number of agents
        hidden_state_size: int
            dim of hidden states per agent (input to policy MLP)
            The input to the value GNN is the *flattened* state across agents.
        action_space_size: int
            action space size per agent
        fc_value_layers: list
             Not directly used as layer sizes, but informs `gnn_value_hidden_size`.
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output (value_support_size)
        gnn_value_hidden_size: int, optional
            Hidden dimension size for the value GNN layers. Defaults to hidden_state_size.
        """
        super().__init__()
        self.num_agents = num_agents
        self.hidden_state_size = hidden_state_size # Per agent
        self.action_space_size = action_space_size

        value_gnn_hidden = gnn_value_hidden_size if gnn_value_hidden_size is not None else hidden_state_size

        # Value function using GraphNetNN
        # Input feature per agent for value prediction: hidden_state
        self.fc_value = GraphNetNN(
            sa_dim=hidden_state_size, # Input is hidden state per agent
            n_agents=num_agents,
            hidden_size=value_gnn_hidden, # Internal hidden dim for GNN
            output_dim=full_support_size, # value_support_size
            pool_type='avg', # Or 'max'
            use_agent_id=False # Assuming value doesn't need agent IDs
        )

        # Policy function using MLP (applied per agent)
        # Input is hidden state per agent
        self.fc_policy = mlp(hidden_state_size, fc_policy_layers, action_space_size, use_value_out=True)

    def forward(self, hidden_state_all_agents):
        # hidden_state_all_agents: [B, N*H] (flattened representation)
        # OR potentially [B, N, H] depending on how it's passed from dynamics/representation
        # Let's assume it's [B, N, H] as that's more natural output from dynamics
        # We need to adapt based on the caller. If caller passes [B, N*H], we reshape.
        # If caller passes [B, N, H], we use it directly for policy and flatten for value.

        batch_size = hidden_state_all_agents.size(0)
        # Infer shape H = hidden_state_size (per agent)
        if hidden_state_all_agents.dim() == 2: # Input is flattened [B, N*H]
            expected_flat_dim = self.num_agents * self.hidden_state_size
            if hidden_state_all_agents.shape[1] != expected_flat_dim:
                 raise ValueError(f"PredictionNetwork input has wrong flat dimension. "
                                 f"Expected {expected_flat_dim}, got {hidden_state_all_agents.shape[1]}.")
            hidden_state_per_agent = hidden_state_all_agents.view(batch_size, self.num_agents, self.hidden_state_size) # [B, N, H]
            hidden_state_flat_for_gnn = hidden_state_all_agents # Already flat for GNN
        elif hidden_state_all_agents.dim() == 3: # Input is [B, N, H]
            if hidden_state_all_agents.shape[1] != self.num_agents or hidden_state_all_agents.shape[2] != self.hidden_state_size:
                 raise ValueError(f"PredictionNetwork input has wrong shape. "
                                 f"Expected [B, {self.num_agents}, {self.hidden_state_size}], "
                                 f"got {hidden_state_all_agents.shape}")
            hidden_state_per_agent = hidden_state_all_agents
            hidden_state_flat_for_gnn = hidden_state_per_agent.reshape(batch_size, -1) # Flatten for GNN input [B, N*H]
        else:
            raise ValueError(f"PredictionNetwork received input with unexpected dimensions: {hidden_state_all_agents.shape}")


        # Value prediction (collective, uses GNN on flattened input)
        value = self.fc_value(hidden_state_flat_for_gnn) # [B, value_support_size]

        # Policy prediction (per agent, uses MLP)
        # Reshape hidden state to process each agent independently [B*N, H]
        policy_input = hidden_state_per_agent.reshape(batch_size * self.num_agents, self.hidden_state_size)
        policy_logit = self.fc_policy(policy_input) # [B*N, Action_Size]
        # Reshape policy logits back to [B, N, Action_Size]
        policy_logit = policy_logit.reshape(batch_size, self.num_agents, self.action_space_size)

        return policy_logit, value


# --- ProjectionNetwork remains the same ---
class ProjectionNetwork(nn.Module):
    def __init__(
        self,
        projection_in_dim: int, # Should be N * H (total hidden state dim)
        proj_hid: int = 256,
        proj_out: int = 256,
        pred_hid: int = 64,
        pred_out: int = 256,
    ):
        super().__init__()
        self.projection_in_dim = projection_in_dim
        # Use the potentially updated mlp implementation
        self.projection = mlp(self.projection_in_dim, [proj_hid], proj_out, use_ReLU=True) # ReLU in projection
        # Norm after projection, before prediction head
        self.projection_norm = nn.LayerNorm(proj_out)
        self.prediction = mlp(proj_out, [pred_hid], pred_out, use_ReLU=True) # ReLU in prediction head

    def project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state expected to be [B, N*H]
        if hidden_state.shape[1] != self.projection_in_dim:
             raise ValueError(f"ProjectionNetwork.project input dim mismatch. "
                              f"Expected {self.projection_in_dim}, got {hidden_state.shape[1]}")
        proj = self.projection(hidden_state)
        proj = self.projection_norm(proj)
        return proj

    def predict(self, projection: torch.Tensor) -> torch.Tensor:
        return self.prediction(projection)


# --- MAMuZeroNet is MODIFIED (only initialization arguments for DynamicsNetwork) ---
class MAMuZeroNet(BaseNet):
    def __init__(
        self,
        num_agents: int,
        observation_shape: Tuple[int, int, int],
        action_space_size: int,
        hidden_state_size: int, # Per agent hidden state size
        fc_representation_layers: List[int],
        fc_dynamic_layers: List[int],
        fc_reward_layers: List[int], # Interpreted as hidden layers for reward GNN if applicable
        fc_value_layers: List[int],  # Interpreted as hidden layers for value GNN if applicable
        fc_policy_layers: List[int],
        reward_support_size: int,
        value_support_size: int,
        inverse_value_transform: Any,
        inverse_reward_transform: Any,
        gnn_value_hidden_size: int = None, # Optional: Hidden size for Value GNN
        gnn_reward_hidden_size: int = None, # Optional: Hidden size for Reward GNN
        proj_hid: int = 256,
        proj_out: int = 256,
        pred_hid: int = 64,
        pred_out: int = 256,
        **kwargs # Catches use_feature_norm etc.
    ):
        super(MAMuZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform)
        self.use_feature_norm = kwargs.get("use_feature_norm", True) # Default True if not provided

        self.num_agents = num_agents
        # Assuming observation_shape is [C, W, H] or similar per agent
        self.obs_size = np.prod(observation_shape) # Flat size per agent
        self.action_space_size = action_space_size # Per agent
        self.hidden_state_size = hidden_state_size # Per agent
        # Total hidden state dimension across all agents
        self.total_hidden_state_size = num_agents * hidden_state_size

        self.representation_network = RepresentationNetwork(
            self.obs_size,
            hidden_state_size, # Output size per agent
            fc_representation_layers,
            self.use_feature_norm
        )

        # Pass the new optional GNN hidden size arguments
        self.dynamics_network = DynamicsNetwork(
            self.num_agents,
            hidden_state_size, # H per agent
            action_space_size, # A per agent
            fc_dynamic_layers,
            fc_reward_layers, # Passed for potential use inside GNN if modified
            reward_support_size,
            gnn_reward_hidden_size=gnn_reward_hidden_size # Pass optional arg
        )

        # Pass the new optional GNN hidden size arguments
        self.prediction_network = PredictionNetwork(
            num_agents,
            hidden_state_size, # H per agent
            action_space_size, # A per agent
            fc_value_layers, # Passed for potential use inside GNN if modified
            fc_policy_layers,
            value_support_size,
            gnn_value_hidden_size=gnn_value_hidden_size # Pass optional arg
        )

        self.projection_network = ProjectionNetwork(
            self.total_hidden_state_size, # Projection input is the combined hidden state
            proj_hid,
            proj_out,
            pred_hid,
            pred_out,
        )

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_state: [B, N*H] (Expected by PredictionNetwork's GNN value head and Projection)
        # Prediction network handles reshaping internally if needed for policy head
        if hidden_state.shape[1] != self.total_hidden_state_size:
            # Maybe it's coming in as [B, N, H]? Check and reshape if necessary
            if hidden_state.dim() == 3 and hidden_state.shape[1] == self.num_agents and hidden_state.shape[2] == self.hidden_state_size:
                hidden_state = hidden_state.view(hidden_state.size(0), -1) # Reshape to [B, N*H]
            else:
                raise ValueError(f"Prediction input shape mismatch. Expected flat dim "
                                 f"{self.total_hidden_state_size}, got {hidden_state.shape}")

        policy_logit, value = self.prediction_network(hidden_state)
        # policy_logit: [B, N, A]
        # value: [B, value_support_size]
        return policy_logit, value

    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        # observation: [B, N, Obs_Features] (e.g., [B, N, C*W*H] after flattening)
        # Or potentially [B, N, C, W, H] - needs flattening first
        batch_size = observation.shape[0]

        # Flatten observation if needed (assuming it comes in per-agent)
        # Example: if observation is [B, N, C, W, H]
        if observation.dim() > 3:
            obs_flat_per_agent = observation.reshape(batch_size * self.num_agents, -1)
        # Example: if observation is already [B, N, Features]
        elif observation.dim() == 3:
            obs_flat_per_agent = observation.reshape(batch_size * self.num_agents, self.obs_size)
        else:
             raise ValueError(f"Representation input shape not recognized: {observation.shape}")

        # Representation network outputs per-agent hidden state [B*N, H]
        hidden_state_per_agent = self.representation_network(obs_flat_per_agent)
        # Reshape to [B, N, H]
        hidden_state_per_agent_reshaped = hidden_state_per_agent.reshape(batch_size, self.num_agents, self.hidden_state_size)

        # Return the concatenated/flattened hidden state [B, N*H]
        # This matches the expected input format for projection and prediction's value head
        hidden_state_total = hidden_state_per_agent_reshaped.view(batch_size, -1)
        return hidden_state_total


    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_state: [B, N*H] (input state)
        # action: [B, N] (indices of actions)
        batch_size = hidden_state.shape[0]

        # Reshape hidden_state to [B, N, H] for dynamics network processing
        hidden_state_per_agent = hidden_state.view(batch_size, self.num_agents, self.hidden_state_size)

        # Convert action indices to one-hot vectors [B, N, A]
        action_onehot = F.one_hot(action.long(), num_classes=self.action_space_size).float()
        # Ensure shape is [B, N, A] even if input action was [B*N]
        if action_onehot.dim() == 2 and action_onehot.shape[0] == batch_size * self.num_agents:
             action_onehot = action_onehot.view(batch_size, self.num_agents, self.action_space_size)
        elif action_onehot.dim() != 3 or action_onehot.shape[1] != self.num_agents:
             raise ValueError(f"Action one-hot encoding failed. Shape: {action_onehot.shape}")


        # Dynamics network takes [B, N, H] and [B, N, A], returns [B, N, H] and [B, reward_support]
        next_hidden_state_per_agent, reward = self.dynamics_network(hidden_state_per_agent, action_onehot)

        # Reshape next_hidden_state back to [B, N*H] for consistency
        next_hidden_state_total = next_hidden_state_per_agent.view(batch_size, -1)

        return next_hidden_state_total, reward # reward is already [B, reward_support]

    def project(self, hidden_state: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        # hidden_state: [B, N*H]
        if hidden_state.shape[1] != self.total_hidden_state_size:
            raise ValueError(f"Project input shape mismatch. Expected flat dim "
                             f"{self.total_hidden_state_size}, got {hidden_state.shape}")

        proj = self.projection_network.project(hidden_state)

        if with_grad:
            # only the branch of proj + pred can share the gradients
            proj = self.projection_network.predict(proj)
            return proj
        else:
            # Return the projected representation without prediction head & grad detached
            return proj.detach()

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        # observation: [B, N, C, W, H] or [B, N, Features]
        # Representation network expects flattened per-agent obs [B*N, Features]
        # and returns [B, N*H]
        hidden_state = self.representation(observation) # [B, N*H]

        # Prediction network takes [B, N*H]
        policy_logit, value_logits = self.prediction(hidden_state)
        # policy_logit: [B, N, A]
        # value_logits: [B, value_support_size]

        # Convert logits to scalar values if not training
        value_scalar = self.inverse_value_transform(value_logits) if not self.training else None
        # Reward is 0 for initial step
        reward_scalar = torch.zeros(observation.size(0), 1, device=observation.device) # Use B from obs

        # Ensure outputs are numpy if not training
        if not self.training:
            hidden_state_np = hidden_state.detach().cpu().numpy()
            # Detach reward scalar as well
            reward_scalar_np = reward_scalar.detach().cpu().numpy()
            value_scalar_np = value_scalar.detach().cpu().numpy()
            policy_logit_np = policy_logit.detach().cpu().numpy()
            # Keep logits for MCTS? MuZero usually uses logits. Let's return logits.
            # Return raw logits/supports during training, scalars otherwise?
            # The base class expects specific types. Let's stick to NetworkOutput definition.
            # It expects hidden_state (Tensor), reward (Tensor/np), value (Tensor/np), policy_logits (Tensor/np)
            # Let's return tensors during training and numpy arrays otherwise.

            return NetworkOutput(hidden_state=hidden_state, # Keep as tensor for potential future recurrent steps
                                 reward=reward_scalar_np,   # Numpy scalar reward
                                 value=value_scalar_np,     # Numpy scalar value
                                 policy_logits=policy_logit_np) # Numpy policy logits
        else:
             # During training, return tensors (specifically the support distributions)
             return NetworkOutput(hidden_state=hidden_state,           # Tensor state
                                  reward=reward_scalar,            # Tensor reward (zeros)
                                  value=value_logits,              # Tensor value support
                                  policy_logits=policy_logit)      # Tensor policy logits


    def recurrent_inference(self, hidden_state: HiddenState, action: Action) -> NetworkOutput:
        # hidden_state: [B, N*H] (Tensor)
        # action: [B, N] (Tensor, action indices)

        # Dynamics network takes [B, N*H] and [B, N] -> returns [B, N*H] and [B, reward_support]
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)

        # Prediction network takes [B, N*H] -> returns [B, N, A] and [B, value_support]
        policy_logit, value_logits = self.prediction(next_hidden_state)

        # Convert logits to scalar values if not training
        reward_scalar = self.inverse_reward_transform(reward_logits) if not self.training else None
        value_scalar = self.inverse_value_transform(value_logits) if not self.training else None

        if not self.training:
            # Numpy outputs for evaluation/acting
             reward_scalar_np = reward_scalar.detach().cpu().numpy()
             value_scalar_np = value_scalar.detach().cpu().numpy()
             policy_logit_np = policy_logit.detach().cpu().numpy()

             return NetworkOutput(hidden_state=next_hidden_state, # Keep tensor state
                                  reward=reward_scalar_np,
                                  value=value_scalar_np,
                                  policy_logits=policy_logit_np)
        else:
            # Tensor outputs (supports) for training
             return NetworkOutput(hidden_state=next_hidden_state,
                                  reward=reward_logits,   # Return reward support
                                  value=value_logits,     # Return value support
                                  policy_logits=policy_logit) # Return policy logits