import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

class aggergator_Graph(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # A simple 2-layer Graph Convolutional Network
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, in_channels)
        self.relu = nn.ReLU()
        self.edge_index = self.get_view_edge_index().cuda()


    def forward(self, x, batch_size, num_views):
        """
        x: Image features of shape [Batch * Num_Views, Channels]
        """
        x = x.cuda()
        # 1. Message Passing: Let views talk to their neighbors
        # We process the whole batch of graphs at once
        # Create a batch vector to keep track of which nodes belong to which object in the batch
        batch_idx = torch.arange(batch_size).repeat_interleave(num_views).to(x.device)
        
        # Repeat the edge_index for each item in the batch
        # This shifts the node indices so graph 2's nodes don't connect to graph 1's nodes
        edge_indices = []
        for i in range(batch_size):
            offset = i * num_views
            edge_indices.append(self.edge_index + offset)
        batched_edge_index = torch.cat(edge_indices, dim=1).cuda()

        # Apply GCN layers
        x = self.conv1(x, batched_edge_index)
        x = self.relu(x)
        x = self.conv2(x, batched_edge_index)
        
        # 2. Aggregation (Pooling)
        # Combine the 10 view nodes into 1 single feature vector per 3D object
        # Global mean pool averages the nodes for each graph in the batch
        aggr_feat = global_mean_pool(x, batch_idx) 
        
        return aggr_feat.cpu()

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