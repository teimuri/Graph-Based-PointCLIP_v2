import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

class aggergator_Graph(nn.Module):
    def __init__(self, in_channels, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        # 1. Graph Attention (GAT) Layers
        # We use multiple heads, dividing in_channels by heads so the output dims remain constant
        self.conv1 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(in_channels)
        
        self.conv2 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True)
        self.norm2 = nn.LayerNorm(in_channels)

        # 2. Final projection layer to merge Max and Mean pooling
        self.fc = nn.Linear(in_channels * 2, in_channels)
        
        # Keep this on CPU initially, we will move it to the correct device in forward()
        self.register_buffer('edge_index', self.get_view_edge_index())

    def get_view_edge_index(self):
        # Assuming you have your edge logic here! 
        # (Replace with your actual implementation)
        pass

    def forward(self, x, batch_size, num_views):
        """
        x: Image features of shape [Batch * Num_Views, Channels]
        """
        # 1. Reshape the flat tensor into [Batch, Num_Views, Channels]
        x_reshaped = x.view(batch_size, num_views, -1)*0
        
        # 2. Sum across the views (dimension 1) to get [Batch, Channels]
        aggr_feat = x_reshaped.sum(dim=1)
        
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