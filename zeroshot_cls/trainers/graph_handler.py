import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torchvision.utils import save_image  # <-- Added for saving image tensors

class aggergator_Graph(nn.Module):
    def __init__(self, in_channels, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        # 1. Graph Attention (GAT) Layers
        self.conv1 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(in_channels)
        
        self.conv2 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True)
        self.norm2 = nn.LayerNorm(in_channels)

        # 2. Final projection layer to merge Max and Mean pooling
        self.fc = nn.Linear(in_channels * 2, in_channels)

    def forward(self, x, batch_size, num_views, images):
        """
        x: Image features of shape [Batch * Num_Views, Channels]
        images: Image tensors of shape [Batch * Num_Views, C, H, W]
        """
        device = x.device
        self.edge_index = []
        
        # Reshape images to group by batch and view without flattening the spatial dims
        C, H, W = images.shape[1], images.shape[2], images.shape[3]
        images_grouped = images.view(batch_size, num_views, C, H, W)
        
        for i, (offset, image_batch) in enumerate(zip(torch.arange(batch_size, device=device), images_grouped)):
            # Pass batch_idx (i) so we don't overwrite images from different batches
            self.edge_index.append(self.get_view_edge_index(image_batch, batch_idx=i) + offset * num_views)
            
        batched_edge_index = torch.cat(self.edge_index, dim=1)
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_views)
        
        # --- 2. Layer 1: GAT + Norm + ReLU + Dropout + Residual ---
        identity = x
        x = self.conv1(x, batched_edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity  
        
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

    def save_graph_images(self, images, batch_idx, folder_name="graph_images"):
        """
        Helper function to save the tensor images to a directory.
        """
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
            
        # images shape here is [num_views, C, H, W]
        for view_idx, img in enumerate(images):
            # Save format: graph_images/batch_0_view_1.png
            file_path = os.path.join(folder_name, f"batch_{batch_idx}_view_{view_idx}.png")
            save_image(img, file_path)

    def get_view_edge_index(self, images, batch_idx=0):
        # 1. Save the images
        self.save_graph_images(images, batch_idx)

        # 2. Define the connections based on geometric proximity
        edges = [
            # Ring connections (forming a circle around the object)
            [4, 0], [0, 5], [5, 1], [1, 6], [6, 2], [2, 7], [7, 3], [3, 4],
            
            # Top camera (9) connects to all ring cameras
            [9, 4], [9, 0], [9, 5], [9, 1], [9, 6], [9, 2], [9, 7], [9, 3],
            
            # Bottom camera (8) connects to all ring cameras
            [8, 4], [8, 0], [8, 5], [8, 1], [8, 6], [8, 2], [8, 7], [8, 3]
        ]
        
        # Add reverse edges for undirected graph
        reverse_edges = [[dst, src] for src, dst in edges]
        all_edges = edges + reverse_edges
        
        # Convert to PyTorch tensor
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        return edge_index.cuda()

    def save_gnn(self, path):
        directory = os.path.dirname(path)
        if directory: 
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)
    
    def load_gnn(self, path):
        self.load_state_dict(torch.load(path))