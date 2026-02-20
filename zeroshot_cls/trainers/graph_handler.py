import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torchvision.utils import save_image
from transformers import AutoImageProcessor, AutoModel

class DinoV2FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "facebook/dinov2-small"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images):
        device = next(self.model.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        features = outputs.pooler_output
        return features

class DynamicGraphBuilder:
    def __init__(self, k):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k

    def build_edges(self, features):
        num_views = features.shape[0]
        device = features.device

        features_norm = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(features_norm, features_norm.t())

        k_for_search = min(self.k + 1, num_views)
        _, top_k_indices = torch.topk(sim_matrix, k=k_for_search, dim=1)

        src_nodes_list, dst_nodes_list = [], []
        for i in range(num_views):
            neighbors = top_k_indices[i, 1:k_for_search]
            src_nodes_list.extend([i] * len(neighbors))
            dst_nodes_list.extend(neighbors.tolist())

        src_nodes = torch.tensor(src_nodes_list, device=device)
        dst_nodes = torch.tensor(dst_nodes_list, device=device)

        edges = torch.stack([src_nodes, dst_nodes])
        all_edges = torch.cat([edges, torch.stack([edges[1], edges[0]])], dim=1)
        edge_index = all_edges.unique(dim=1)

        return edge_index.contiguous()

class aggergator_Graph(nn.Module):
    def __init__(self, in_channels, heads=4, dropout=0.2, graph_mode='dynamic', k_neighbors=8):
        super().__init__()
        self.dropout = dropout
        self.graph_mode = graph_mode
        
        self.conv1 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv2 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True)
        self.norm2 = nn.LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels * 2, in_channels)

        if self.graph_mode == 'dynamic':
            if k_neighbors <= 0:
                raise ValueError("k_neighbors must be a positive integer for dynamic mode.")
            self.feature_extractor = DinoV2FeatureExtractor()
            self.graph_builder = DynamicGraphBuilder(k=k_neighbors)

    def forward(self, x, batch_size, num_views, images, save_image_flag=False):
        device = x.device
        edge_indices = []

        images_grouped = images.view(batch_size, num_views, *images.shape[1:])

        for i, image_batch_per_object in enumerate(images_grouped):
            offset = i * num_views
            
            if self.graph_mode == 'dynamic':
                view_features = self.feature_extractor(image_batch_per_object)
                edge_index = self.graph_builder.build_edges(view_features)
            else:
                edge_index = self.get_static_view_edge_index(device)
            
            edge_indices.append(edge_index + offset)

        batched_edge_index = torch.cat(edge_indices, dim=1)
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
        
    def get_static_view_edge_index(self, device):
        edges = [
            [4, 0], [0, 5], [5, 1], [1, 6], [6, 2], [2, 7], [7, 3], [3, 4],
            [9, 4], [9, 0], [9, 5], [9, 1], [9, 6], [9, 2], [9, 7], [9, 3],
            [8, 4], [8, 0], [8, 5], [8, 1], [8, 6], [8, 2], [8, 7], [8, 3]
        ]
        reverse_edges = [[dst, src] for src, dst in edges]
        all_edges = edges + reverse_edges
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        return edge_index.to(device)

    def save_gnn(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_gnn(self, path):
        self.load_state_dict(torch.load(path))
