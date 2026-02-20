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
        
        # بسیار مهم: تعریف edge_dim=1 برای دریافت امتیاز هندسی Zernike در یال‌ها
        self.conv1 = GATConv(in_channels, in_channels // heads, heads=heads, edge_dim=1, concat=True)
        self.norm1 = nn.LayerNorm(in_channels)
        
        self.conv2 = GATConv(in_channels, in_channels // heads, heads=heads, edge_dim=1, concat=True)
        self.norm2 = nn.LayerNorm(in_channels)

        # لایه نهایی برای ترکیب Poolingها
        self.fc = nn.Linear(in_channels * 2, in_channels)

    def _get_zernike_features(self, img_np, radius=32, degree=8):
        """ استخراج ویژگی‌های Zernike برای مقاوم‌سازی در برابر نویز ScanObjectNN """
        _, mask = cv2.threshold(img_np, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 3) # حذف نویزهای نقطه‌ای اسکن
        
        # پیدا کردن مرکز ثقل برای مقاومت در برابر انتقال
        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        else:
            cx, cy = img_np.shape[1] // 2, img_np.shape[0] // 2

        # محاسبه ویژگی‌های Zernike
        features = mahotas.features.zernike_moments(mask, radius, degree, cm=(cy, cx))
        return features

    def get_static_edges_with_zernike_weights(self, images, device):
        """ ساخت یال‌های استاتیک و محاسبه وزن هندسی Zernike برای هر یال """
        num_views = images.shape[0]
        
        # ۱. تعریف لبه‌های استاتیک (همان‌هایی که نتیجه 40.9 دادند)
        # مثال: Ring + Top + Bottom
        static_edges = [
            [4, 0], [0, 5], [5, 1], [1, 6], [6, 2], [2, 7], [7, 3], [3, 4], # Ring
            [9, 4], [9, 0], [9, 5], [9, 1], [9, 6], [9, 2], [9, 7], [9, 3], # Top to Ring
            [8, 4], [8, 0], [8, 5], [8, 1], [8, 6], [8, 2], [8, 7], [8, 3]  # Bottom to Ring
        ]

        # ۲. تبدیل تصاویر به NumPy و محاسبه ویژگی‌های Zernike
        images_np_raw = images.cpu().numpy()
        if images_np_raw.shape[1] > 1: # اگر RGB بود
            images_grayscale = np.mean(images_np_raw, axis=1)
        else:
            images_grayscale = np.squeeze(images_np_raw, axis=1)
        
        images_np = (images_grayscale * 255).astype(np.uint8)
        z_feats = [self._get_zernike_features(img) for img in images_np]

        final_edges = []
        final_attrs = []

        for src, dst in static_edges:
            # محاسبه شباهت هندسی بر اساس فاصله Zernike
            dist = np.linalg.norm(z_feats[src] - z_feats[dst])
            weight = np.exp(-dist) # تبدیل فاصله به امتیاز شباهت (بین 0 و 1)

            # یال رفت و برگشت به همراه وزن یکسان
            final_edges.append([src, dst])
            final_attrs.append([weight])
            final_edges.append([dst, src])
            final_attrs.append([weight])

        edge_index = torch.tensor(final_edges, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(final_attrs, dtype=torch.float, device=device)
        
        return edge_index, edge_attr

    def forward(self, x, batch_size, num_views, images, save_image):
        """
        x: ویژگی‌های CLIP با ابعاد [Batch * Num_Views, Channels]
        images: تصاویر ورودی با ابعاد [Batch * Num_Views, C, H, W]
        """
        device = x.device
        all_edge_index = []
        all_edge_attr = []

        # گروه بندی تصاویر بر اساس بچ
        C, H, W = images.shape[1], images.shape[2], images.shape[3]
        images_grouped = images.view(batch_size, num_views, C, H, W)
        
        # مرحله اول: ساخت گراف و استخراج وزن‌های هندسی برای هر بچ
        for i in range(batch_size):
            offset = i * num_views
            image_batch = images_grouped[i]
            
            # دریافت ایندکس یال‌ها و ویژگی یال‌ها (Zernike Weights)
            edge_index, edge_attr = self.get_static_edges_with_zernike_weights(image_batch, device)
            
            all_edge_index.append(edge_index + offset)
            all_edge_attr.append(edge_attr)
            
        batched_edge_index = torch.cat(all_edge_index, dim=1)
        batched_edge_attr = torch.cat(all_edge_attr, dim=0) # ابعاد [Total_Edges, 1]
        
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_views)

        # --- لایه اول GAT: ترکیب اطلاعات CLIP (نود) و Zernike (یال) ---
        identity = x
        x = self.conv1(x, batched_edge_index, edge_attr=batched_edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity # Residual connection
        
        # --- لایه دوم GAT ---
        identity = x
        x = self.conv2(x, batched_edge_index, edge_attr=batched_edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity # Residual connection

        # --- Pooling نهایی برای به دست آوردن یک بردار برای هر آبجکت ---
        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        
        # ترکیب Mean و Max و کاهش بعد به ابعاد اصلی کانال‌ها
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
        print(f"DEBUG: Trying to save model to {path}...")
        directory = os.path.dirname(path)
        if directory: 
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)
        print("DEBUG: Save successful!")
    
    def load_gnn(self, path):
        checkpoint = torch.load(path)
        print("Keys in loaded file:", checkpoint.keys())
        self.load_state_dict(torch.load(path))

