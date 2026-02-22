import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torchvision.utils import save_image
import numpy as np
import math # <-- تغییر 1: ایمپورت کردن کتابخانه math

# ==============================================================================
#  کلاس کمکی برای محاسبه ممان‌های زرنیک به صورت کامل روی GPU
# ==============================================================================
class ZernikeMomentsGPU:
    def __init__(self, height, width, degree, device):
        self.device = device
        self.basis = self._precompute_zernike_basis(height, width, degree).to(device)
        self.n_moments = self.basis.shape[0]

    def _precompute_zernike_basis(self, height, width, degree):
        """
        پایه‌های چندجمله‌ای زرنیک را یک بار پیش‌محاسبه می‌کند.
        """
        Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        Y = (2.0 * Y - height + 1) / height
        X = (2.0 * X - width + 1) / width
        
        rho = torch.sqrt(X**2 + Y**2)
        theta = torch.atan2(Y, X)
        
        mask = rho <= 1.0
        
        basis_functions = []
        n_coeffs = 0
        
        # --- تغییر 2: استفاده از math.factorial به جای np.math.factorial ---
        factorials = torch.tensor([math.factorial(i) for i in range(degree + 1)], dtype=torch.float32)

        for n in range(degree + 1):
            for m in range(n + 1):
                if (n - m) % 2 == 0:
                    radial_poly = torch.zeros_like(rho)
                    for k in range((n - m) // 2 + 1):
                        numerator = (-1)**k * factorials[n - k]
                        denominator = factorials[k] * factorials[(n + 2*k - m) // 2] * factorials[(n - 2*k - m) // 2]
                        radial_poly += (numerator / denominator) * (rho**(n - 2 * k))
                    
                    if m == 0:
                        zernike_poly = torch.sqrt(torch.tensor(n + 1.0)) * radial_poly
                        basis_functions.append(zernike_poly * mask)
                    else:
                        zernike_poly_cos = torch.sqrt(torch.tensor(2.0 * (n + 1.0))) * radial_poly * torch.cos(m * theta)
                        zernike_poly_sin = torch.sqrt(torch.tensor(2.0 * (n + 1.0))) * radial_poly * torch.sin(m * theta)
                        basis_functions.append(zernike_poly_cos * mask)
                        basis_functions.append(zernike_poly_sin * mask)
                        
        return torch.stack(basis_functions)

    def __call__(self, silhouettes):
        """
        ممان‌ها را برای یک بچ از سیلوئت‌ها محاسبه می‌کند.
        """
        area = silhouettes.sum(dim=[1, 2], keepdim=True)
        area[area == 0] = 1
        norm_silhouettes = silhouettes / area

        moments = torch.einsum('bxy,mxy->bm', norm_silhouettes, self.basis)
        return torch.abs(moments)


# ==============================================================================
#  کلاس اصلی گراف با استفاده از محاسبه‌گر GPU (بدون تغییر)
# ==============================================================================
class aggergator_Graph(nn.Module):
    def __init__(self, in_channels, heads=4, dropout=0.2, image_size=224, zernike_degree=8):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True, edge_dim=1)
        self.norm1 = nn.LayerNorm(in_channels)
        
        self.conv2 = GATConv(in_channels, in_channels // heads, heads=heads, concat=True, edge_dim=1)
        self.norm2 = nn.LayerNorm(in_channels)
        
        self.fc = nn.Linear(in_channels * 2, in_channels)

        self.zernike_calculator = None
        self.image_size = image_size
        self.zernike_degree = zernike_degree

        self.register_buffer('gray_weights', torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1))
        
        edges = [
            [4, 0], [0, 5], [5, 1], [1, 6], [6, 2], [2, 7], [7, 3], [3, 4],
            [9, 4], [9, 0], [9, 5], [9, 1], [9, 6], [9, 2], [9, 7], [9, 3],
            [8, 4], [8, 0], [8, 5], [8, 1], [8, 6], [8, 2], [8, 7], [8, 3]
        ]
        reverse_edges = [[dst, src] for src, dst in edges]
        all_edges = edges + reverse_edges
        self.register_buffer('static_edge_index', torch.tensor(all_edges, dtype=torch.long).t().contiguous())

    def forward(self, x, batch_size, num_views, images, save_image):
        device = x.device
        
        if self.zernike_calculator is None:
            self.zernike_calculator = ZernikeMomentsGPU(
                self.image_size, self.image_size, self.zernike_degree, device
            )

        all_edge_indices = []
        all_edge_attrs = []

        images_grouped = images.view(batch_size, num_views, *images.shape[1:])
        
        for i, (offset, image_batch) in enumerate(zip(torch.arange(batch_size, device=device), images_grouped)):
            edge_index, edge_attr = self.get_edges_and_attributes_gpu(image_batch)
            all_edge_indices.append(edge_index + offset * num_views)
            all_edge_attrs.append(edge_attr)
            
        batched_edge_index = torch.cat(all_edge_indices, dim=1)
        batched_edge_attr = torch.cat(all_edge_attrs, dim=0).unsqueeze(1)

        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_views)
        
        identity = x
        x = self.conv1(x, batched_edge_index, edge_attr=batched_edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity
        
        identity = x
        x = self.conv2(x, batched_edge_index, edge_attr=batched_edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity

        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        
        aggr_feat = torch.cat([x_mean, x_max], dim=1)
        aggr_feat = self.fc(aggr_feat)
        
        return aggr_feat

    def get_edges_and_attributes_gpu(self, images_batch):
        gray_images = (images_batch * self.gray_weights).sum(dim=1)
        silhouettes = (gray_images < (245 / 255.0)).float()
        moments = self.zernike_calculator(silhouettes)
        edge_index = self.static_edge_index
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        
        zernike_src = moments[src_nodes]
        zernike_dst = moments[dst_nodes]
        
        dist = F.pairwise_distance(zernike_src, zernike_dst, p=2)
        similarity = torch.exp(-dist)

        return edge_index, similarity

    def save_gnn(self, path):
        directory = os.path.dirname(path)
        if directory: 
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_gnn(self, path):
        self.load_state_dict(torch.load(path))

