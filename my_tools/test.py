# import torch
# import torch.nn as nn


# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module '''
#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
#         self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

#         self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

#     def forward(self, q, k, v, mask=None):
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#         sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
#         residual = q

#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

#         if mask is not None:
#             mask = mask.unsqueeze(1)   # For head axis broadcasting.
#         q, attn = self.attention(q, k, v, mask=mask)

#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
#         q = self.dropout(self.fc(q))
#         q += residual

#         q = self.layer_norm(q)

#         return q, attn
  
# query = torch.rand(128, 32, 256)
# multihead_attn = MultiHeadAttention(n_head=8, d_model=256, d_k=32, d_v=32)
# attn_output, attn_weights = multihead_attn(query, query, query)
# print(f'attn_output: {attn_output.size()}, attn_weights: {attn_weights.size()}')

# query = torch.rand(128, 32, 256)
# multihead_attn = MultiHeadAttention(n_head=8, d_model=256, d_k=256, d_v=512)
# attn_output, attn_weights = multihead_attn(query, query, query)
# print(f'attn_output: {attn_output.size()}, attn_weights: {attn_weights.size()}')

import numpy as np
import cv2

def generate_gaussian_kernel(size, sigma):
    """生成二维高斯核"""
    kx = cv2.getGaussianKernel(size, sigma)
    ky = cv2.getGaussianKernel(size, sigma)
    kernel = np.multiply(kx, np.transpose(ky))
    return kernel

def apply_gaussian_to_points(feature_map, points, kernel_size=21, sigma=3):
    """基于harris角点提取显著区域"""
    height, width = feature_map.shape[2], feature_map.shape[3]
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    half_size = kernel_size // 2

    saliency_map = np.zeros((height, width))

    for point in points:
        x = int(point[0] * width)
        y = int(point[1] * height)
        # 在 (x, y) 位置应用高斯核
        x_min = max(x - half_size, 0)
        x_max = min(x + half_size + 1, width)
        y_min = max(y - half_size, 0)
        y_max = min(y + half_size + 1, height)
        
        kx_min = half_size - (x - x_min)
        kx_max = half_size + (x_max - x)
        ky_min = half_size - (y - y_min)
        ky_max = half_size + (y_max - y)
        
        saliency_map[y_min:y_max, x_min:x_max] += kernel[ky_min:ky_max, kx_min:kx_max]

    return saliency_map

# 示例特征图
B, C, H, W = 1, 3, 56, 56
feature_map = np.random.rand(B, C, H, W)

# 已获得的harris角点位置 (B, N, X, Y)
harris_points = np.array([[[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]]])  # 示例数据

# 提取显著区域
saliency_map = apply_gaussian_to_points(feature_map, harris_points[0], kernel_size=21, sigma=3)

# 显示结果
import matplotlib.pyplot as plt

plt.imshow(saliency_map, cmap='hot')
plt.colorbar()
plt.show()
plt.savefig('fg1.png')

