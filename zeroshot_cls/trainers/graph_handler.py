import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torchvision.utils import save_image
import cv2  # <-- کتابخانه OpenCV برای پردازش تصویر اضافه شد
import numpy as np  # <-- کتابخانه NumPy برای کار با آرایه‌ها اضافه شد

import mahotas


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

    def forward(self, x, batch_size, num_views, images, save_image):
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
            edges = self.get_view_edge_index(image_batch, batch_idx=i, save_image=save_image)
            self.edge_index.append(edges + offset * num_views)
            
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

    def _get_hu_moments_score(self, image_np_1, image_np_2):
        """
        یک تابع کمکی برای محاسبه امتیاز شباهت بین دو تصویر با استفاده از مومنت‌های هو.
        امتیاز کمتر به معنای شباهت بیشتر است.
        """
        # آستانه‌گذاری برای ساخت ماسک باینری از شیء
        _, binary_mask1 = cv2.threshold(image_np_1, 10, 255, cv2.THRESH_BINARY)
        _, binary_mask2 = cv2.threshold(image_np_2, 10, 255, cv2.THRESH_BINARY)
        
        # محاسبه امتیاز شباهت با استفاده از تابع داخلی OpenCV
        score = cv2.matchShapes(binary_mask1, binary_mask2, cv2.CONTOURS_MATCH_I1, 0.0)
        return score

    def _get_zernike_features(self, img_np, radius=21, degree=8):
        """
        استخراج ویژگی‌های Zernike برای یک تصویر.
        degree=8 حدود 25 ویژگی بسیار دقیق استخراج می‌کند.
        """
        # ۱. پیش‌پردازش برای حذف نویز ScanObjectNN
        _, mask = cv2.threshold(img_np, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 3) # حذف نویزهای نقطه‌ای اسکن
        
        # ۲. پیدا کردن مرکز ثقل (Centroid) برای مقاوم کردن نسبت به انتقال
        # Zernike باید حول مرکز ثقل محاسبه شود
        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = img_np.shape[1] // 2, img_np.shape[0] // 2

        # ۳. محاسبه مومنت‌های Zernike
        # این تابع برداری از ویژگی‌های ارتوگونال برمی‌گرداند
        features = mahotas.features.zernike_moments(mask, radius, degree, cm=(cy, cx))
        return features

    def get_view_edge_index(self, images, batch_idx=0, save_image=False):
        num_views = images.shape[0]
        device = images.device
        K = 2 

        # تبدیل تنسورها به NumPy
        images_np_raw = images.cpu().numpy()
        if images_np_raw.shape[1] > 1:
            images_grayscale = np.mean(images_np_raw, axis=1)
        else:
            images_grayscale = np.squeeze(images_np_raw, axis=1)
        
        images_np = (images_grayscale * 255).astype(np.uint8)

        # ۱. استخراج ویژگی‌های Zernike برای تمام ویوها (Pre-calculation)
        # استفاده از degree=8 به ما 25 ویژگی قدرتمند می‌دهد (بسیار بیشتر از 7 عدد Hu)
        zernike_feats = []
        for i in range(num_views):
            feat = self._get_zernike_features(images_np[i], radius=32, degree=8)
            zernike_feats.append(feat)
        
        zernike_feats = np.array(zernike_feats) # تبدیل به ماتریس [num_views, num_features]

        # ۲. محاسبه ماتریس فاصله (Euclidean Distance)
        # برخلاف Hu، اینجا مستقیماً فاصله اقلیدسی بین بردارها معنای فیزیکی و هندسی دارد
        dist_matrix = np.zeros((num_views, num_views))
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # محاسبه فاصله اقلیدسی بین بردار ویژگی ویو i و j
                d = np.linalg.norm(zernike_feats[i] - zernike_feats[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # ۳. ساخت گراف KNN بر اساس کمترین فاصله
        all_edges = []
        for i in range(num_views):
            # پیدا کردن K همسایه نزدیک (به جز خودش)
            # مقادیر را مرتب می‌کنیم و ایندکس‌ها را برمی‌داریم
            closest_neighbors = np.argsort(dist_matrix[i])[1:K+1]
            for neighbor_idx in closest_neighbors:
                all_edges.append([i, neighbor_idx])

        # ۴. تبدیل به فرمت PyTorch Geometric
        if not all_edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        edge_index = torch.tensor(all_edges, dtype=torch.long, device=device).t().contiguous()
        
        # اطمینان از غیرجهت‌دار بودن گراف (Undirected)
        from torch_geometric.utils import to_undirected
        edge_index = to_undirected(edge_index)
        
        return edge_index

    def save_graph_images(self, images, batch_idx, folder_name="graph_images"):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
            
        for view_idx, img in enumerate(images):
            file_path = os.path.join(folder_name, f"batch_{batch_idx}_view_{view_idx}.png")
            save_image(img, file_path)
        raise ValueError("End of images saving")

    def save_gnn(self, path):
        directory = os.path.dirname()
        if directory: 
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)
    
    def load_gnn(self, path):
        self.load_state_dict(torch.load(path))

