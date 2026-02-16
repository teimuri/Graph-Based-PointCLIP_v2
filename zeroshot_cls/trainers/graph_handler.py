import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class aggergator_Graph(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        # The GNN acts purely as a scoring mechanism
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Projects the hidden graph features into a single scalar weight per view
        self.scorer = nn.Linear(hidden_dim, 1)
        self.register_buffer('edge_index', self.get_view_edge_index())

    def get_view_edge_index(self):
        # Your edge index logic here
        pass

    def forward(self, x, batch_size, num_views):
        device = x.device
        
        # Shift edge indices for the batch
        edge_offset = (torch.arange(batch_size, device=device) * num_views).view(-1, 1, 1)
        batched_edge_index = (self.edge_index.unsqueeze(0) + edge_offset).transpose(0, 1).reshape(2, -1)
        
        # --- 1. Graph Message Passing (To learn view importance) ---
        h = F.relu(self.conv1(x, batched_edge_index))
        h = F.relu(self.conv2(h, batched_edge_index))
        
        # Calculate raw scores for each view
        raw_scores = self.scorer(h).view(batch_size, num_views, 1)
        
        # Softmax ensures the weights for each object's 10 views sum to 1.0
        attn_weights = F.softmax(raw_scores, dim=1)
        
        # --- 2. Weighted Sum of the ORIGINAL Features ---
        x_reshaped = x.view(batch_size, num_views, -1)
        aggr_feat = (x_reshaped * attn_weights).sum(dim=1)
        
        return aggr_feat

    def get_view_edge_index(self):
        # Define the connections based on geometric proximity
        # Format: [source_node, target_node]
        edges = [
            # Ring connections (forming a circle around the object)
            [4, 0], [0, 5], [5, 1], [1, 6], [6, 2], [2, 7], [7, 3], [3, 4],
            
            # Top camera (9) connects to all ring cameras
            [9, 4], [9, 0], [9, 5], [9, 1], [9, 6], [9, 2], [9, 7], [9, 3],
            
            # Bottom camera (8) connects to all ring cameras
            [8, 4], [8, 0], [8, 5], [8, 1], [8, 6], [8, 2], [8, 7], [8, 3]
        ]
        
        # Graphs are typically undirected in this context, so we add the reverse edges
        reverse_edges = [[dst, src] for src, dst in edges]
        all_edges = edges + reverse_edges
        
        # Convert to PyTorch tensor of shape [2, num_edges]
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        return edge_index