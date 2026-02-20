import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, knn_graph
from torch_geometric.utils import to_undirected

class aggergator_Graph(nn.Module):
    def __init__(self, in_channels, heads=4, dropout=0.2, k_neighbors=2):

        super().__init__()
        self.dropout = dropout
        self.k_neighbors = k_neighbors 

        self.conv1 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(in_channels)

        self.conv2 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True)
        self.norm2 = nn.LayerNorm(in_channels)
        
        self.fc = nn.Linear(in_channels * 2, in_channels)

    def build_dynamic_graph(self, view_features):

        edge_index = knn_graph(view_features, k=self.k_neighbors, loop=True, cosine=True)
        edge_index = to_undirected(edge_index)
        return edge_index.to(view_features.device)

    def forward(self, x, batch_size, num_views, images, save_image=False):

        device = x.device
        all_edge_indices = []

        for i in range(batch_size):
            start_idx = i * num_views
            end_idx = (i + 1) * num_views
            current_view_features = x[start_idx:end_idx]

            edge_index = self.build_dynamic_graph(current_view_features)

            all_edge_indices.append(edge_index + start_idx)

        batched_edge_index = torch.cat(all_edge_indices, dim=1)
        
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_views)

        identity = x
        x = self.conv1(x, batched_edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity

        identity = x
        x = self.conv2(x, batched_edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity

        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)

        aggr_feat = torch.cat([x_mean, x_max], dim=1)
        aggr_feat = self.fc(aggr_feat)

        return aggr_feat

    def save_graph_images(self, images, batch_idx, folder_name="graph_images"):

        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
            
        for view_idx, img in enumerate(images):
            file_path = os.path.join(folder_name, f"batch_{batch_idx}_view_{view_idx}.png")
            save_image(img, file_path)
    
    def save_gnn(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_gnn(self, path):
        self.load_state_dict(torch.load(path))

