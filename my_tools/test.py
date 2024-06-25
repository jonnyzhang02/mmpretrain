import cv2
import torch
import numpy as np

def harris_corner_detection_and_topk_corners(feature_maps, k=0.04, block_size=2, ksize=3, top_k=16):
    """
    Harris corner detection on a batch of feature maps using OpenCV and PyTorch,
    and return the top K corners for each feature map with normalized coordinates.

    Args:
        feature_maps: Input feature maps (B, C, H, W).
        k: Harris detector free parameter.
        block_size: It is the size of neighbourhood considered for corner detection.
        ksize: Aperture parameter of Sobel derivative used.
        top_k: Number of top corners to return.

    Returns:
        corners_list: List of coordinates of the top K corners for each image in the batch.
    """
    batch_size, num_channels, height, width = feature_maps.shape
    corners_list = []

    for i in range(batch_size):
        # Convert the feature map to a single channel by averaging across channels
        feature_map = feature_maps[i].mean(dim=0).cpu().numpy()

        # Ensure feature_map is in the right type
        feature_map = np.float32(feature_map)

        # Detecting corners using OpenCV
        dst = cv2.cornerHarris(feature_map, block_size, ksize, k)

        # Result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Convert the response back to a PyTorch tensor
        harris_response = torch.tensor(dst, dtype=torch.float32, device=feature_maps.device)

        # Extract top K corners
        topk_values, topk_indices = torch.topk(harris_response.view(-1), top_k)
        topk_coords = torch.stack((torch.div(topk_indices, width, rounding_mode='floor'), topk_indices % width), dim=-1)

        # Normalize the coordinates
        topk_coords = topk_coords.float()
        topk_coords[:, 0] /= height
        topk_coords[:, 1] /= width

        # Append the normalized top K corners for the current feature map
        corners_list.append(topk_coords)



    # Stack the list of top K corners into a single tensor
    corners_list = torch.stack(corners_list, dim=0)
    
    return corners_list

# 示例使用
B, C, H, W = 4, 3, 100, 100
feature_maps = torch.rand(B, C, H, W)

# 进行 Harris 角点检测并提取顶点的16个角点（归一化坐标）
topk_corners = harris_corner_detection_and_topk_corners(feature_maps)
print(topk_corners.shape)

# # 显示第一个特征图的 Harris 角点响应和顶点的16个角点位置（归一化后）
# import matplotlib.pyplot as plt

# harris_response_np = topk_corners[0].cpu().numpy()
# # print(harris_response_np)

# plt.imshow(feature_maps[0].mean(dim=0).cpu(), cmap='gray')
# plt.scatter(harris_response_np[:, 1] * W, harris_response_np[:, 0] * H, c='red', s=10)
# plt.title('Harris Corner Detection with Top 16 Normalized Corners')
# plt.show()
