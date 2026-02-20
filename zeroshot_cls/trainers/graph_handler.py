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
    def __init__(self, in_channels, heads=4, dropout=0.5): # Dropout رو بالا بردیم برای ریسک
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GATConv(in_channels, in_channels // heads, heads=heads, edge_dim=1, concat=True)
        self.norm1 = nn.LayerNorm(in_channels)
        
        self.conv2 = GATConv(in_channels, in_channels // heads, heads=heads, edge_dim=1, concat=True)
        self.norm2 = nn.LayerNorm(in_channels)

        self.fc = nn.Linear(in_channels * 2, in_channels)

    def _get_zernike_features(self, img_np):
        """ استخراج ویژگی زرنیک (همان متد قبلی با فیلتر نویز) """
        _, mask = cv2.threshold(img_np, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 3)
        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        else:
            cx, cy = img_np.shape[1]//2, img_np.shape[0]//2
        return mahotas.features.zernike_moments(mask, radius=32, degree=8, cm=(cy, cx))


    def get_random_geometric_edges(self, images, device, num_random_edges=20):
        """ 
        فوت کوزه‌گری: ساخت یال‌های تصادفی اما با وزن هندسی 
        """
        num_random_edges = torch.randint(low=5, high=12, size=(1,)).item()

        num_views = images.shape[0]
        images_cpu = images.cpu().numpy()
        images_grayscale = images_cpu[:, 0, :, :] 
        # ۱. استخراج ویژگی‌های زرنیک
        images_np = (images_grayscale * 255).astype(np.uint8)
        z_feats = [self._get_zernike_features(img) for img in images_np]

        # ۲. تولید یال‌های تصادفی (Random Sampling of Edges)
        # ما اجازه می‌دیم هر نودی به هر نودی وصل بشه، اما به صورت تصادفی انتخاب می‌کنیم
        all_possible_pairs = torch.combinations(torch.arange(num_views), r=2)
        # انتخاب تصادفی تعدادی از جفت‌ها در هر ران
        perm = torch.randperm(all_possible_pairs.size(0))[:num_random_edges]
        selected_pairs = all_possible_pairs[perm]

        final_edges = []
        final_attrs = []

        for src, dst in selected_pairs:
            src, dst = src.item(), dst.item()
            # محاسبه شباهت زرنیک
            dist = np.linalg.norm(z_feats[src] - z_feats[dst])
            weight = np.exp(-dist / 0.5) # وزن هندسی

            # اضافه کردن یال دوطرفه
            final_edges.append([src, dst])
            final_attrs.append([weight])
            final_edges.append([dst, src])
            final_attrs.append([weight])

        edge_index = torch.tensor(final_edges, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(final_attrs, dtype=torch.float, device=device)
        
        return edge_index, edge_attr

    def forward(self, x, batch_size, num_views, images, save_image):
        device = x.device
        if self.training:
            noise = torch.randn_like(x, device=device)
            scales = torch.full((batch_size * num_views, 1), 0.01, device=device)
            for b in range(batch_size):
                corrupted_indices = torch.randperm(num_views)[:3]
                scales[b * num_views + corrupted_indices] = 0.2
            x = x + noise * scales
            x = F.normalize(x, p=2, dim=-1)
        all_edge_index = []
        all_edge_attr = []

        images_grouped = images.view(batch_size, num_views, images.shape[1], images.shape[2], images.shape[3])
        
        for i in range(batch_size):
            offset = i * num_views
            # قمار اصلی: یال‌های هر نمونه در هر Batch کاملاً تصادفی و متفاوت است
            edge_index, edge_attr = self.get_random_geometric_edges(images_grouped[i], device)
            
            all_edge_index.append(edge_index + offset)
            all_edge_attr.append(edge_attr)
            
        batched_edge_index = torch.cat(all_edge_index, dim=1)
        batched_edge_attr = torch.cat(all_edge_attr, dim=0)
        
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_views)

        # اجرای GNN با تکیه بر یال‌های تصادفی اما وزن‌دار
        identity = x
        x = self.conv1(x, batched_edge_index, edge_attr=batched_edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x + identity 
        
        x = self.conv2(x, batched_edge_index, edge_attr=batched_edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = x + identity 

        aggr_feat = torch.cat([global_mean_pool(x, batch_idx), global_max_pool(x, batch_idx)], dim=1)
        return self.fc(aggr_feat)
    
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

