import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import os
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
        # self.register_buffer('edge_index', )

    def forward(self, x, batch_size, num_views, images):
        """
        x: Image features of shape [Batch * Num_Views, Channels]
        """
        device = x.device
        self.edge_index = []
        for offset,image_batch in zip(torch.arange(batch_size, device=device),images.view(batch_size,num_views,-1)):
            self.edge_index.append(self.get_view_edge_index(image_batch)+offset*num_views)
        batched_edge_index = torch.cat(self.edge_index,dim=1)
        
        # --- 2. Layer 1: GAT + Norm + ReLU + Dropout + Residual ---
        identity = x
        x = self.conv1(x, batched_edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity  # Residual skip connection
        
        # --- 3. Layer 2: GAT + Norm + ReLU + Dropout + Residual ---
        identity = x
        x = self.conv2(x, batched_edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity 

        # --- 4. Rich Aggregation (Mean + Max Pooling) ---
        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        
        # Concatenate and project back to original channel dimension
        aggr_feat = torch.cat([x_mean, x_max], dim=1)
        aggr_feat = self.fc(aggr_feat)
        
        return aggr_feat

    def get_view_edge_index(self, images):
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
        return edge_index.cuda()

    def save_gnn(self,path):
        # 1. Extract the directory part of the path (e.g., "saved_models/exp_1")
        directory = os.path.dirname(path)
        
        # 2. If a directory was specified, create it (and any parent folders)
        if directory: 
            os.makedirs(directory, exist_ok=True)
            
        # 3. Now it is safe to save!
        torch.save(self.state_dict(), path)
    
    def load_gnn(self, path):
        self.load_state_dict(torch.load(path))