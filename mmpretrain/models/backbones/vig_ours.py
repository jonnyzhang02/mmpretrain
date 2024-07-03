# Copyright (c) OpenMMLab. All rights reserved.
# modified from
# https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import ModuleList, Sequential
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mmpretrain.registry import MODELS
from ..utils import build_norm_layer
import cv2
import matplotlib.pyplot as plt


def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed,
                                 pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def xy_pairwise_distance(x, y):
    """Compute pairwise distance of a point cloud.

    Args:
        x: tensor (batch_size, num_points, num_dims)
        y: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.

    Args:
        x: (batch_size, num_dims, num_points, 1)
        y: (batch_size, num_dims, num_points, 1)
        k: int
        relative_pos:Whether to use relative_pos
    Returns:
        nearest neighbors:
        (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(
            0, n_points, device=x.device).repeat(batch_size, k,
                                                 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)

def harris_corner_detection_and_topk_corners(feature_maps, vi=0, k=0.04, block_size=2, ksize=3, top_k=16, threshold=0.01):
    """
    Improved Harris corner detection with thresholding, non-maximum suppression, and optional visualization.
    """
    batch_size, num_channels, height, width = feature_maps.shape
    corners_list = []

    for i in range(batch_size):
        feature_map = feature_maps[i].mean(dim=0).cpu().detach().numpy()
        feature_map = np.float32(feature_map)
        dst = cv2.cornerHarris(feature_map, block_size, ksize, k)
        dst = cv2.dilate(dst, None)

        # Apply threshold to the Harris response
        _, dst_thresh = cv2.threshold(dst, threshold * dst.max(), 255, 0)
        dst_thresh = dst_thresh.astype(np.uint8)
        corners = cv2.goodFeaturesToTrack(dst_thresh, maxCorners=top_k, qualityLevel=0.01, minDistance=5)

        # Normalize and convert to tensor
        if corners is not None:
            corners = torch.tensor(corners, dtype=torch.float32, device=feature_maps.device).squeeze(1)
            corners[:, 0] /= width
            corners[:, 1] /= height
        else:
            corners = torch.zeros((0, 2), dtype=torch.float32, device=feature_maps.device)

        # 保证top_k个角点
        if corners.size(0) < top_k:
            # Randomly sample with replacement
            indices = torch.randint(0, corners.size(0), (top_k - corners.size(0),), device=feature_maps.device)
            additional_corners = corners[indices]
            corners = torch.cat([corners, additional_corners], dim=0)

        corners_list.append(corners)

    if vi:
        corners = corners_list[0].cpu()
        # Visualization of corners on the feature map
        plt.figure(figsize=(10, 10))
        plt.imshow(feature_maps[0].mean(dim=0).cpu().detach().numpy(), cmap='gray')
        if corners is not None:
            corners_unnorm = corners.clone()
            corners_unnorm[:, 0] *= width
            corners_unnorm[:, 1] *= height
            plt.scatter(corners_unnorm[:, 0], corners_unnorm[:, 1], c='r', s=100, marker='x')
        plt.savefig('harris.png')

    corners_list = torch.stack(corners_list, dim=0)
    return corners_list

# def harris_corner_detection_and_topk_corners(feature_maps, k=0.04, block_size=2, ksize=3, top_k=16):
#     """
#     Harris corner detection on a batch of feature maps using OpenCV and PyTorch,
#     and return the top K corners for each feature map with normalized coordinates.

#     Args:
#         feature_maps: Input feature maps (B, C, H, W).
#         k: Harris detector free parameter.
#         block_size: It is the size of neighbourhood considered for corner detection.
#         ksize: Aperture parameter of Sobel derivative used.
#         top_k: Number of top corners to return.

#     Returns:
#         corners_list: (B, num_points, 2)
#     """
#     batch_size, num_channels, height, width = feature_maps.shape
#     corners_list = []

#     for i in range(batch_size):
#         # Convert the feature map to a single channel by averaging across channels
#         feature_map = feature_maps[i].mean(dim=0).cpu().detach().numpy()

#         # Ensure feature_map is in the right type
#         feature_map = np.float32(feature_map)

#         # Detecting corners using OpenCV
#         dst = cv2.cornerHarris(feature_map, block_size, ksize, k)

#         # Result is dilated for marking the corners, not important
#         dst = cv2.dilate(dst, None)

#         # Convert the response back to a PyTorch tensor
#         harris_response = torch.tensor(dst, dtype=torch.float32, device=feature_maps.device)

#         # Extract top K corners
#         topk_values, topk_indices = torch.topk(harris_response.view(-1), top_k)
#         topk_coords = torch.stack((torch.div(topk_indices, width, rounding_mode='floor'), topk_indices % width), dim=-1)

#         # Normalize the coordinates
#         topk_coords = topk_coords.float()
#         topk_coords[:, 0] /= height
#         topk_coords[:, 1] /= width

#         # Append the normalized top K corners for the current feature map
#         corners_list.append(topk_coords)

#         # # Visualization
#         # visualize_corners(feature_map, topk_coords, height, width)

#     # Stack the list of top K corners into a single tensor
#     corners_list = torch.stack(corners_list, dim=0)
    
#     return corners_list

def visualize_corners(feature_map, corners, height, width):
    """
    Visualize corners on a feature map.

    Args:
        feature_map: A single channel feature map.
        corners: Normalized coordinates of the corners (num_points, 2).
        height: Height of the feature map.
        width: Width of the feature map.
    """
    corners = corners.cpu()
    plt.figure(figsize=(8, 8))
    plt.imshow(feature_map, cmap='gray')
    corners_denorm = corners.clone().detach()
    corners_denorm[:, 0] *= height
    corners_denorm[:, 1] *= width
    plt.scatter(corners_denorm[:, 1], corners_denorm[:, 0], c='r', s=40)
    plt.title("Harris Corners Visualization")
    plt.show()
    plt.savefig('harris.png')

def calculate_harris_corner_loss(tensor1, tensor2):
    B = tensor1.shape[0]
    num_points = tensor1.shape[1]
    tensor1_flattened = tensor1.view(B, num_points*2)
    tensor2_flattened = tensor2.view(B, num_points*2)

    # 计算每个batch内的余弦相似度
    return F.cosine_similarity(tensor1_flattened, tensor2_flattened, dim=1)

def visualize_feature_map(tensor, figname, batch_idx=0, channel_idx=0):
    """
    可视化特征图的函数。
    
    参数:
        tensor (torch.Tensor): 输入的四维tensor，维度为 [B, C, H, W]。
        batch_idx (int): 选择要可视化的批次索引。
        channel_idx (int): 选择要可视化的通道索引，长度为3或1。
    
    返回:
        None
    """

    # 提取单通道图像并删除通道维度
    img = tensor[batch_idx, channel_idx, :, :].detach().cpu().numpy()
    img = np.expand_dims(img, axis=-1)  # 为了显示灰度图，扩展维度

    # 归一化图像以获得更好的视觉效果
    img -= img.min()
    img /= img.max()

    # 显示图像
    plt.imshow(img, cmap='gray' if img.shape[-1] == 1 else None)
    plt.title(f'Batch {batch_idx}, Channels {channel_idx}')
    plt.savefig(figname)



class DenseDilated(nn.Module):
    """Find dilated neighbor from neighbor list.

    edge_index: (2, batch_size, num_points, k)
    """

    def __init__(self, k=9, dilation=1, use_stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.use_stochastic = use_stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.use_stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """Find the neighbors' indices based on dilated knn."""

    def __init__(self, k=9, dilation=1, use_stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.use_stochastic = use_stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, use_stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)

            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation,
                                             relative_pos)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            y = x.clone()

            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation,
                                             relative_pos)
        return self._dilated(edge_index)


class BasicConv(Sequential):

    def __init__(self,
                 channels,
                 act_cfg,
                 norm_cfg=None,
                 graph_conv_bias=True,
                 drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(
                nn.Conv2d(
                    channels[i - 1],
                    channels[i],
                    1,
                    bias=graph_conv_bias,
                    groups=4))
            if norm_cfg is not None:
                m.append(build_norm_layer(norm_cfg, channels[-1]))
            if act_cfg is not None:
                m.append(build_activation_layer(act_cfg))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)


def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:
                `\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(
        0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced,
                                  -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k,
                           num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class MRConv2d(nn.Module):
    """Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)
    for dense data type."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg,
                 norm_cfg=None,
                 graph_conv_bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act_cfg, norm_cfg,
                            graph_conv_bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)],
                      dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """Edge convolution layer (with activation, batch normalization) for dense
    data type."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg,
                 norm_cfg=None,
                 graph_conv_bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act_cfg, norm_cfg,
                            graph_conv_bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(
            self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216)
    for dense data type."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg,
                 norm_cfg=None,
                 graph_conv_bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act_cfg, norm_cfg,
                             graph_conv_bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act_cfg,
                             norm_cfg, graph_conv_bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for
    dense data type."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg,
                 norm_cfg=None,
                 graph_conv_bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act_cfg, norm_cfg,
                            graph_conv_bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """Static graph convolution layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 graph_conv_type,
                 act_cfg,
                 norm_cfg=None,
                 graph_conv_bias=True):
        super(GraphConv2d, self).__init__()
        if graph_conv_type == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act_cfg,
                                    norm_cfg, graph_conv_bias)
        elif graph_conv_type == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act_cfg, norm_cfg,
                                  graph_conv_bias)
        elif graph_conv_type == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act_cfg,
                                   norm_cfg, graph_conv_bias)
        elif graph_conv_type == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act_cfg,
                                   norm_cfg, graph_conv_bias)
        else:
            raise NotImplementedError(
                'graph_conv_type:{} is not supported'.format(graph_conv_type))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """Dynamic graph convolution layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 k=9,
                 dilation=1,
                 graph_conv_type='mr',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=None,
                 graph_conv_bias=True,
                 use_stochastic=False,
                 epsilon=0.2,
                 r=1):
        super(DyGraphConv2d,
              self).__init__(in_channels, out_channels, graph_conv_type,
                             act_cfg, norm_cfg, graph_conv_bias)
        self.k = k
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(k, dilation,
                                                      use_stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """Grapher module with graph convolution and fc layers."""

    def __init__(self,
                 in_channels,
                 k=9,
                 dilation=1,
                 graph_conv_type='mr',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=None,
                 graph_conv_bias=True,
                 use_stochastic=False,
                 epsilon=0.2,
                 r=1,
                 n=196,
                 drop_path=0.0,
                 relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, k,
                                        dilation, graph_conv_type, act_cfg,
                                        norm_cfg, graph_conv_bias,
                                        use_stochastic, epsilon, r)
        self.fc2 = Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), in_channels),
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.relative_pos = None
        if relative_pos:
            relative_pos_tensor = torch.from_numpy(
                np.float32(
                    get_2d_relative_pos_embed(in_channels, int(
                        n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor,
                size=(n, n // (r * r)),
                mode='bicubic',
                align_corners=False)
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(
                relative_pos.unsqueeze(0), size=(N, N_reduced),
                mode='bicubic').squeeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class FFN(nn.Module):
    """"out_features = out_features or in_features\n
        hidden_features = hidden_features or in_features"""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), hidden_features),
        )
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), out_features),
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x
    

class SaliencyExtractor(nn.Module):
    def __init__(self, kernel_size_factor=0.1, sigma=3):
        super(SaliencyExtractor, self).__init__()
        self.kernel_size_factor = kernel_size_factor
        self.sigma = sigma

    def generate_gaussian_kernel(self, size, sigma):
        """生成二维高斯核"""
        kx = cv2.getGaussianKernel(size, sigma)
        ky = cv2.getGaussianKernel(size, sigma)
        kernel = np.outer(kx, ky)
        return torch.tensor(kernel, dtype=torch.float32)

    def apply_gaussian_to_points(self, feature_map, points):
        """基于harris角点提取显著区域"""
        B, C, height, width = feature_map.shape
        kernel_size = self.determine_kernel_size(min(height, width))
        kernel = self.generate_gaussian_kernel(kernel_size, self.sigma)
        half_size = kernel_size // 2

        saliency_maps = torch.zeros((B, height, width), dtype=torch.float32)

        for b in range(B):
            for point in points[b]:
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
                
                saliency_maps[b, y_min:y_max, x_min:x_max] += kernel[ky_min:ky_max, kx_min:kx_max]

        return saliency_maps

    def determine_kernel_size(self, feature_map_size):
        """根据特征图的尺寸确定高斯核大小"""
        size = int(feature_map_size * self.kernel_size_factor)
        return size if size % 2 == 1 else size + 1

    def forward(self, feature_map, points):
        self.saliency_maps = self.apply_gaussian_to_points(feature_map, points) 
        self.visual()
        return self.apply_gaussian_to_points(feature_map, points)   

    def visual(self):
        # 将显著性图转换为numpy以便可视化
        saliency_map_np = self.saliency_maps[0].detach().numpy()

        # 显示结果
        plt.imshow(saliency_map_np, cmap='hot')
        plt.colorbar()
        plt.show() 
        plt.savefig('gauss.png')
    
class TargetEnhanceModule(nn.Module):
    def __init__(self):
        super(TargetEnhanceModule, self).__init__()
    
    def forward(self, x, saliency_ex):
        O_obj = x * saliency_ex

        return O_obj


class BackgroundMixModule(nn.Module):
    def __init__(self):
        super(BackgroundMixModule, self).__init__()

    def forward(self, x, saliency_ex):
        O_env = x * (1 - saliency_ex)

        return O_env


@MODELS.register_module()
class PyramidVigOurs(BaseBackbone):
    """Pyramid Vision GNN backbone.

    A PyTorch implementation of `Vision GNN: An Image is Worth Graph of Nodes
    <https://arxiv.org/abs/2206.00272>`_.

    Modified from the official implementation
    https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch

    Args:
        arch (str): Vision GNN architecture, choose from 'tiny',
            'small' and 'base'.
        in_channels (int): The number of channels of input images.
            Defaults to 3.
        k (int): The number of KNN's k. Defaults to 9.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        act_cfg (dict): The config of activative functions.
            Defaults to ``dict(type='GELU'))``.
        norm_cfg (dict): The config of normalization layers.
            Defaults to ``dict(type='BN')``.
        graph_conv_bias (bool): Whether to use bias in the convolution
            layers in Grapher. Defaults to True.
        graph_conv_type (str): The type of graph convolution，choose
            from 'edge', 'mr', 'sage' and 'gin'. Defaults to 'mr'.
        epsilon (float): Probability of random arrangement in KNN. It only
            works when ``use_stochastic=True``. Defaults to 0.2.
        use_stochastic (bool): Whether to use stochastic in KNN.
            Defaults to False.
        drop_path (float): stochastic depth rate. Default 0.0
        norm_eval (bool): Whether to set the normalization layer to eval mode.
            Defaults to False.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        init_cfg (dict, optional): The initialization configs.
            Defaults to None.
    """  # noqa: E501
    arch_settings = {
        # 'tiny': dict(blocks=[2, 2, 6, 2], channels=[48, 96, 240, 384]),
        'tiny' : dict(blocks=[2, 2, 6, 2], channels=[48, 96, 120, 192]),
        'small': dict(blocks=[2, 2, 6, 2], channels=[80, 160, 400, 640]),
        'medium': dict(blocks=[2, 2, 16, 2], channels=[96, 192, 384, 768]),
        'base': dict(blocks=[2, 2, 18, 2], channels=[128, 256, 512, 1024]),
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 k=9,
                 out_indices=-1,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN'),
                 graph_conv_bias=True,
                 graph_conv_type='mr',
                 epsilon=0.2,
                 use_stochastic=False,
                 drop_path=0.,
                 norm_eval=False,
                 frozen_stages=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        arch = self.arch_settings[arch]
        self.blocks = arch['blocks']
        self.num_blocks = sum(self.blocks)
        self.num_stages = len(self.blocks)
        channels = arch['channels']
        self.channels = channels

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_stages + index
            assert 0 <= out_indices[i] <= self.num_stages, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.stem = Sequential(
            nn.Conv2d(in_channels, channels[0] // 2, 3, stride=2, padding=1),
            build_norm_layer(norm_cfg, channels[0] // 2),
            build_activation_layer(act_cfg),
            nn.Conv2d(channels[0] // 2, channels[0], 3, stride=2, padding=1),
            build_norm_layer(norm_cfg, channels[0]),
            build_activation_layer(act_cfg),
            nn.Conv2d(channels[0], channels[0], 3, stride=1, padding=1),
            build_norm_layer(norm_cfg, channels[0]),
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]
        # number of knn's k
        num_knn = [
            int(x.item()) for x in torch.linspace(k, k, self.num_blocks)
        ]
        max_dilation = 49 // max(num_knn)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, channels[0], 224 // 4, 224 // 4))
        HW = 224 // 4 * 224 // 4
        reduce_ratios = [4, 2, 1, 1]

        self.stages = ModuleList()
        block_idx = 0
        for stage_idx, num_blocks in enumerate(self.blocks):
            mid_channels = channels[stage_idx]
            reduce_ratio = reduce_ratios[stage_idx]
            blocks = []
            if stage_idx > 0:
                blocks.append(
                    Sequential(
                        nn.Conv2d(
                            self.channels[stage_idx - 1],
                            mid_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1),
                        build_norm_layer(norm_cfg, mid_channels),
                    ))
                HW = HW // 4

            
            for _ in range(num_blocks):
                blocks.append(
                    Sequential(
                        Grapher(
                            in_channels=mid_channels,
                            k=num_knn[block_idx],
                            dilation=min(block_idx // 4 + 1, max_dilation),
                            graph_conv_type=graph_conv_type,
                            act_cfg=act_cfg,
                            norm_cfg=norm_cfg,
                            graph_conv_bias=graph_conv_bias,
                            use_stochastic=use_stochastic,
                            epsilon=epsilon,
                            r=reduce_ratio,
                            n=HW,
                            drop_path=dpr[block_idx],
                            relative_pos=True),
                        FFN(in_features=mid_channels,
                            hidden_features=mid_channels * 4,
                            act_cfg=act_cfg,
                            drop_path=dpr[block_idx])))
                block_idx += 1
            self.stages.append(Sequential(*blocks))

        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

    def forward(self, inputs):
        outs = []
        x = self.stem(inputs) + self.pos_embed
        # visualize_feature_map(inputs, 'input.png')
        # visualize_feature_map(x, 'x.png')
        harris_loss = calculate_harris_corner_loss(
            harris_corner_detection_and_topk_corners(inputs), 
            harris_corner_detection_and_topk_corners(x)
            ).mean()
        saliency_ex = SaliencyExtractor()
        target_enhance_module = TargetEnhanceModule()
        background_mix_module = BackgroundMixModule()
        

        for i, blocks in enumerate(self.stages):
            saliency_map = saliency_ex(x, harris_corner_detection_and_topk_corners(x))
            x_target_enhanced = target_enhance_module(x, saliency_map)
            x_background_mixed = background_mix_module(x, saliency_map)

            x = torch.concat(x_target_enhanced, x_background_mixed, dim=1)

            x = blocks(x)

            # if i in self.out_indices:
            outs.append(x)

        # ############ attention module ##############

        # target_size = outs[0].size()[2:]  # (H, W)
        # # 将所有特征图调整到相同的大小
        # resized_outs = [F.interpolate(out, size=target_size, mode='bilinear', align_corners=False) for out in outs]
        # # 在通道维度上拼接
        # concatenated = torch.cat(resized_outs, dim=1)

        # B, C, H, W = concatenated.size()
        # reshaped_features = concatenated.view(B, C, H*W)
        # B, C_f, H_f , W_f = outs[-1].size()
        # reshaped_features_f = outs[-1].view(B, C_f, H_f*W_f)


        # # 生成位置编码
        # position_embedding = generate_position_embedding(H, W, C)
        # position_embedding_f = generate_position_embedding(H_f, W_f, C_f)
        # # 添加位置编码
        # embedded_features = reshaped_features + position_embedding
        # embedded_features_f = reshaped_features_f + position_embedding_f

        # # 注意力计算
        # attention = torch.matmul(embedded_features.transpose(1, 2), 
        #                          embedded_features_f)
        # attention = F.softmax(attention, dim=-1)
        # # 加权和
        # attended_features = torch.matmul(attention, embedded_features_f.transpose(1, 2))
        # # 重塑输出
        # attended_features = attended_features.view(B, C, H, W)

        return (harris_loss, [outs[-1]])

    def _freeze_stages(self):
        self.stem.eval()
        for i in range(self.frozen_stages):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(PyramidVigOurs, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
