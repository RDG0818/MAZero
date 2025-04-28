import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List # Make sure to import List
from core.graph_layers import GraphConvLayer, MessageFunc, UpdateFunc

class GraphNet(nn.Module):
    """
    A graph net for permutation-invariant value estimation.
    Takes agent features and outputs a single value prediction.
    """
    def __init__(self, input_feature_dim, n_agents, hidden_size, output_dim,
                 pool_type='avg', use_agent_id_features=False, agent_id_feature_dim=2):
        super(GraphNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.pool_type = pool_type
        self.use_agent_id_features = use_agent_id_features
        self.agent_id_feature_dim = agent_id_feature_dim if use_agent_id_features else 0

        current_input_dim = self.input_feature_dim + self.agent_id_feature_dim

        self.gc1 = GraphConvLayer(current_input_dim, hidden_size)
        self.nn_gc1 = nn.Linear(current_input_dim, hidden_size)

        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.nn_gc2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, output_dim) # Output dim matches full_support_size
        # Initialize V weights/biases potentially close to zero as in original mlp
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        # Assumes a fully connected graph. Normalized adjacency.
        adj = (torch.ones(n_agents, n_agents) - torch.eye(n_agents))
        adj = adj / (self.n_agents - 1) # Normalize by degree (N-1 for fully connected excluding self)
        self.register_buffer('adj', adj)

        if self.use_agent_id_features:
            # Simple agent ID embedding (e.g., one-hot or learned)
            # Example: Learned embedding per agent index
            self.agent_embeddings = nn.Embedding(n_agents, self.agent_id_feature_dim)
            # Or fixed one-hot (adjust agent_id_feature_dim to n_agents)
            # self.register_buffer('agent_ids_one_hot', F.one_hot(torch.arange(n_agents)).float())


    def forward(self, x):
        """
        :param x: Agent features, expected shape [batch_size, self.input_feature_dim, self.n_agents]
        :return: Value prediction, shape [batch_size, self.output_dim]
        """
        batch_size = x.shape[0]

        if self.use_agent_id_features:
             # Create agent ID features for the batch
             # Example using learned embeddings:
             agent_ids = torch.arange(self.n_agents, device=x.device)
             id_feats = self.agent_embeddings(agent_ids) # [n_agents, agent_id_feature_dim]
             id_feats = id_feats.permute(1, 0) # [agent_id_feature_dim, n_agents]
             id_feats = id_feats.unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size, agent_id_feature_dim, n_agents]
             # Example using fixed one-hot:
             # id_feats = self.agent_ids_one_hot.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1) #[B, N, N] -> [B, N, N]

             x = torch.cat([x, id_feats], dim=1) # Concatenate along feature dimension

        # Ensure adj is compatible with batch size if GraphConvLayer expects [B, N, N]
        # If GraphConvLayer handles [N, N] adj, this is fine.
        # If it needs [B, N, N], use: adj_batch = self.adj.unsqueeze(0).repeat(batch_size, 1, 1)
        adj_batch = self.adj # Assuming GraphConvLayer handles broadcast or uses [N,N]

        # Permute input for nn.Linear layers (expect features last)
        # x shape: [B, C, N] -> permute -> [B, N, C]
        x_permuted = x.permute(0, 2, 1)

        # Graph Layer 1 + Skip Connection
        feat_gcn1 = F.relu(self.gc1(x, adj_batch))
        feat_lin1 = F.relu(self.nn_gc1(x_permuted)) # input [B, N, C_in] -> output [B, N, H]
        feat_lin1 = feat_lin1.permute(0, 2, 1) # permute back -> [B, H, N]
        feat = feat_gcn1 + feat_lin1
        # Optional normalization (original code had / n_agents, could use LayerNorm)
        # feat = F.layer_norm(feat, feat.shape[1:]) # Normalize over C, N dims
        # feat = feat / 2.0 # Alternative simple normalization if adding

        # Permute for next linear layer
        feat_permuted = feat.permute(0, 2, 1) # [B, N, H]

        # Graph Layer 2 + Skip Connection
        out_gcn2 = F.relu(self.gc2(feat, adj_batch))
        out_lin2 = F.relu(self.nn_gc2(feat_permuted)) # input [B, N, H] -> output [B, N, H]
        out_lin2 = out_lin2.permute(0, 2, 1) # permute back -> [B, H, N]
        out = out_gcn2 + out_lin2
        # Optional normalization
        # out = F.layer_norm(out, out.shape[1:])
        # out = out / 2.0

        # --- Pooling ---
        # out shape: [batch_size, hidden_size, n_agents]
        # Pool over the agent dimension (dim=2)
        if self.pool_type == 'avg':
            pooled_out = out.mean(dim=2)  # [batch_size, hidden_size]
        elif self.pool_type == 'max':
            pooled_out, _ = out.max(dim=2) # [batch_size, hidden_size]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Final layer to get value prediction
        value = self.V(pooled_out) # [batch_size, output_dim]
        return value