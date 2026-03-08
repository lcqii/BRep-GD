import numpy as np 
import math
import torch
import torch.nn as nn
import random
import string
import argparse
from chamferdist import ChamferDistance
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Optional, Tuple, Union
import pickle

from OCC.Core.gp import gp_Pnt, gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from collections import defaultdict

def compute_boxx(surfPos, scale_factor=1.5, MIN_VALUE=-3., MAX_VALUE=3.,epsilon=0.1,thin_threshold=0.01):
    batch_size, length, _ = surfPos.shape

    # 拆分surfPos为两个点的坐标
    point1 = surfPos[:, :, :3]  # [batch_size, length, 3]
    point2 = surfPos[:, :, 3:]  # [batch_size, length, 3]

    # 计算每个框的中心点
    center = (point1 + point2) / 2.0  # [batch_size, length, 3]

    # 计算放大后的点1和点2
    new_point1 = center + (point1 - center) * scale_factor  # [batch_size, length, 3]
    new_point2 = center + (point2 - center) * scale_factor  # [batch_size, length, 3]

    min_point = torch.min(new_point1, new_point2)  # [batch, length, 3]
    max_point = torch.max(new_point1, new_point2)  # [batch, length, 3]

    # 重新计算 box_dims（放缩后新的长宽高）
    box_dims = torch.abs(max_point - min_point)  # [batch, length, 3] -> 放缩后的新尺寸

    # 判断是否是很薄的box，按照高宽长比例判断
    is_thin = (torch.min(box_dims, dim=-1).values / torch.max(box_dims, dim=-1).values) < thin_threshold  # [batch, length]

    # 对于薄的box，单独处理最小点和最大点，在最薄方向扩展 epsilon 或 0.1倍第二长长度
    if is_thin.any():
        # 找出每个维度的最小值、第二大值、最大值
        sorted_dims, _ = torch.sort(box_dims, dim=-1, descending=True)  # [batch, length, 3]
        longest_dims = sorted_dims[:, :, 0]  # 最长的长度 [batch, length]
        second_longest_dims = sorted_dims[:, :, 1]  # 第二长的长度 [batch, length]

        # 扩展规则：默认 epsilon，如果 epsilon > 最长长度，则扩展为 0.1 * 第二长的长度
        effective_epsilon = torch.where(epsilon*2 > longest_dims, 0.2 * second_longest_dims, 0.1)  # [batch, length]

        thin_dims = torch.argmin(box_dims, dim=-1)  # 找出最薄方向的维度（0, 1, 2 表示 x, y, z） [batch, length]

        # 只在最薄的方向上扩展 effective_epsilon
        for i in range(3):
            # thin_in_dim 是 [batch, length]，我们需要将其扩展为 [batch, length, 1]
            thin_in_dim_expanded = (thin_dims == i).unsqueeze(-1)  # [batch, length, 1]
            
            # 确保 effective_epsilon 的维度为 [batch, length, 1]
            effective_epsilon_expanded = effective_epsilon.unsqueeze(-1)  # [batch, length, 1]
            
            # 使用掩码来扩展相应维度的 min_point 和 max_point
            min_point[..., i] = torch.where(thin_in_dim_expanded, min_point[..., i:i+1] - effective_epsilon_expanded, min_point[..., i:i+1]).squeeze(-1)
            max_point[..., i] = torch.where(thin_in_dim_expanded, max_point[..., i:i+1] + effective_epsilon_expanded, max_point[..., i:i+1]).squeeze(-1)
    # min_point = torch.min(new_point1, new_point2)  # [length, 3]
    # max_point = torch.max(new_point1, new_point2)  # [length, 3]

    # # 增加 epsilon 扩展包围框
    # min_point = min_point - epsilon  # 扩展最小点
    # max_point = max_point + epsilon  # 扩展最大点

    min_point = torch.clamp(min_point, min=MIN_VALUE, max=MAX_VALUE)
    max_point = torch.clamp(max_point, min=MIN_VALUE, max=MAX_VALUE)

    # 扩展维度以方便相互比较
    min_point_expanded = min_point.unsqueeze(2)  # [length, 1, 3]
    max_point_expanded = max_point.unsqueeze(2)  # [length, 1, 3]
    min_point_expanded2 = min_point.unsqueeze(1)  # [1, length, 3]
    max_point_expanded2 = max_point.unsqueeze(1)  # [1, length, 3]

    # 计算每个框的最小和最大坐标
    min1 = min_point_expanded  # [length, 1, 3]
    max1 = max_point_expanded  # [length, 1, 3]
    min2 = min_point_expanded2  # [1, length, 3]
    max2 = max_point_expanded2  # [1, length, 3]

    # # 扩展维度以方便相互比较
    # new_point1_expanded = new_point1.unsqueeze(2)  # [batch_size, length, 1, 3]
    # new_point2_expanded = new_point2.unsqueeze(2)  # [batch_size, length, 1, 3]
    # new_point1_expanded2 = new_point1.unsqueeze(1)  # [batch_size, 1, length, 3]
    # new_point2_expanded2 = new_point2.unsqueeze(1)  # [batch_size, 1, length, 3]

    # # 计算每个框的最小和最大坐标
    # min1 = torch.min(new_point1_expanded, new_point2_expanded)  # [batch_size, length, 1, 3]
    # max1 = torch.max(new_point1_expanded, new_point2_expanded)  # 同上
    # min2 = torch.min(new_point1_expanded2, new_point2_expanded2)  # [batch_size, 1, length, 3]
    # max2 = torch.max(new_point1_expanded2, new_point2_expanded2)  # 同上

    # 判断是否相交（在三个轴上都有交集）
    overlaps = (max1 >= min2) & (min1 <= max2)  # [batch_size, length, length, 3]
    intersection = overlaps.all(dim=-1).float()  # [batch_size, length, length]

    # 消除自相交的情况
    eye = torch.eye(length, device=surfPos.device).unsqueeze(0)  # [1, length, length]
    intersection = intersection * (1 - eye)

    # 构建邻接矩阵，初始化为零
    adj = torch.zeros(batch_size, length, length, 7, device=surfPos.device)

    # 第一维设置为相交情况
    adj[..., 0] = intersection  # [batch_size, length, length]

    # 相交框的两点坐标
    adj[..., 1:4] = torch.max(min1, min2) * intersection.unsqueeze(-1)  # [batch_size, length, length, 3]
    adj[..., 4:7] = torch.min(max1, max2) * intersection.unsqueeze(-1)  # 同上

    return adj

def calculate_metrics(link_logits, val_labels, threshold=0.5):
    """
    计算模型在验证集上的 Precision, Recall, F1-score
    
    Args:
    - model: 训练好的链接预测模型
    - val_node_features: 验证集上的节点特征，形状为 (batch_size, n, dim)
    - val_labels: 验证集上的真实连接标签，形状为 (batch_size, n, n, 1)
    - threshold: 判定连接的概率阈值，默认为0.5
    
    Returns:
    - precision: 精确率
    - recall: 召回率
    - f1_score: F1分数
    """
    with torch.no_grad():  # 禁用梯度计算
        link_probs = torch.sigmoid(link_logits)
        # 将概率转换为二进制预测标签
        preds = (link_probs > threshold).float()  # 预测的二值标签 (batch_size, n, n, 1)
        
        # True Positives (TP): preds = 1, labels = 1
        tp = ((preds == 1) & (val_labels == 1)).sum().item()
        
        # False Positives (FP): preds = 1, labels = 0
        fp = ((preds == 1) & (val_labels == 0)).sum().item()
        
        # False Negatives (FN): preds = 0, labels = 1
        fn = ((preds == 0) & (val_labels == 1)).sum().item()

        # 计算精确率 (Precision)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # 计算召回率 (Recall)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 计算 F1-score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
    return precision, recall, f1_score


def generate_random_string(length):
    characters = string.ascii_letters + string.digits  # You can include other characters if needed
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def get_bbox_norm(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return np.linalg.norm(max_point - min_point)


def compute_bbox_center_and_size(min_corner, max_corner):
    # Calculate the center
    center_x = (min_corner[0] + max_corner[0]) / 2
    center_y = (min_corner[1] + max_corner[1]) / 2
    center_z = (min_corner[2] + max_corner[2]) / 2
    center = np.array([center_x, center_y, center_z])
    # Calculate the size
    size_x = max_corner[0] - min_corner[0]
    size_y = max_corner[1] - min_corner[1]
    size_z = max_corner[2] - min_corner[2]
    size = max(size_x, size_y, size_z)
    return center, size


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def pad_repeat(x, max_len):
    repeat_times = math.floor(max_len/len(x))
    sep = max_len-repeat_times*len(x)
    sep1 = np.repeat(x[:sep], repeat_times+1, axis=0)
    sep2 = np.repeat(x[sep:], repeat_times, axis=0)
    x_repeat = np.concatenate([sep1, sep2], 0)
    return x_repeat


def pad_zero(x, max_len, return_mask=False):
    keys = np.ones(len(x))
    padding = np.zeros((max_len-len(x))).astype(int)  
    mask = 1-np.concatenate([keys, padding]) == 1  
    padding = np.zeros((max_len-len(x), *x.shape[1:]))
    x_padded = np.concatenate([x, padding], axis=0)
    if return_mask:
        return x_padded, mask
    else:
        return x_padded


def plot_3d_bbox(ax, min_corner, max_corner, color='r'):
    """
    Helper function for plotting 3D bounding boxese
    """
    vertices = [
        (min_corner[0], min_corner[1], min_corner[2]),
        (max_corner[0], min_corner[1], min_corner[2]),
        (max_corner[0], max_corner[1], min_corner[2]),
        (min_corner[0], max_corner[1], min_corner[2]),
        (min_corner[0], min_corner[1], max_corner[2]),
        (max_corner[0], min_corner[1], max_corner[2]),
        (max_corner[0], max_corner[1], max_corner[2]),
        (min_corner[0], max_corner[1], max_corner[2])
    ]
    # Define the 12 triangles composing the box
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]], 
        [vertices[0], vertices[1], vertices[5], vertices[4]], 
        [vertices[2], vertices[3], vertices[7], vertices[6]], 
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors='blue', linewidths=1, edgecolors=color, alpha=0))
    return


def get_args_vae():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_process/deepcad_parsed', 
                        help='Path to data folder')  
    parser.add_argument('--train_list', type=str, default='data_process/deepcad_data_split_6bit_surface.pkl', 
                        help='Path to training list')  
    parser.add_argument('--val_list', type=str, default='data_process/deepcad_data_split_6bit.pkl', 
                        help='Path to validation list')  
    # Training parameters
    parser.add_argument("--option", type=str, choices=['surface', 'edge'], default='surface', 
                        help="Choose between option surface or edge (default: surface)")
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')   #512
    parser.add_argument('--train_nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--save_nepoch', type=int, default=20, help='number of epochs to save model')
    parser.add_argument('--test_nepoch', type=int, default=10, help='number of epochs to test model')
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--finetune",  action='store_true', help='Finetune from existing weights')
    parser.add_argument("--weight",  type=str, default=None, help='Weight path when finetuning')  
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU IDs to use for training (default: [0])")
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="surface_vae", help='environment')
    parser.add_argument('--dir_name', type=str, default="proj_log", help='name of the log folder.')
    args = parser.parse_args()
    # saved folder
    args.save_dir = f'{args.dir_name}/{args.env}'
    
    # args.data = "data_process/furniture_parsed"
    # args.train_list = "data_process/furniture_data_split_6bit_edge.pkl"
    # args.val_list = "data_process/furniture_data_split_6bit.pkl"
    # args.option = "edge"
    # args.gpu = 0
    # args.env = "furniture_vae_edge"
    # args.train_nepoch = 200
    # args.finetune = True
    # args.weight = "checkpoints/furniture_vae_edge.pt"



    return args


def get_args_ldm():
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_process/deepcad_parsed', 
                        help='Path to data folder')  
    parser.add_argument('--list', type=str, default='data_process/deepcad_data_split_6bit.pkl', 
                        help='Path to data list')  
    parser.add_argument('--surfvae', type=str, default='proj_log/deepcad_surfvae/epoch_400.pt', 
                        help='Path to pretrained surface vae weights')  
    parser.add_argument('--edgevae', type=str, default='proj_log/deepcad_edgevae/epoch_300.pt', 
                        help='Path to pretrained edge vae weights')  
    parser.add_argument("--option", type=str, choices=['ADJ','GEdge','GEvpAC','GEvp','GEdgePosSim','surfpos', 'surfz', 'edgepos', 'edgez','full','GeoEdge','total','GEdgePos','GEdgeZ','PreEdge'], default='total', 
                        help="Choose between option [surfpos,edgepos,surfz,edgez] (default: surfpos)")
    parser.add_argument("--beta_schedule", type=str, choices=['linear','scaled_linear','squaredcos_cap_v2','sigmoid'], default='linear',help="Choose beta schedule type (default: linear)")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')  #512
    parser.add_argument('--train_nepoch', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--test_nepoch', type=int, default=25, help='number of epochs to test model')
    parser.add_argument('--save_nepoch', type=int, default=50, help='number of epochs to save model')
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces') # 50
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face') # 30
    parser.add_argument('--threshold', type=float, default=0.05, help='minimum threshold between two faces') #0.05
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    parser.add_argument('--z_scaled', type=float, default=1, help='scaled the latent z')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0, 1], help="GPU IDs to use for training (default: [0, 1])")
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--cf",  action='store_true', help='Use data augmentation')
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="surface_pos", help='environment')
    parser.add_argument('--dir_name', type=str, default="proj_log", help='name of the log folder.')
    #args = parser.parse_args()
    args, _ = parser.parse_known_args()
    # saved folder
    args.save_dir = f'{args.dir_name}/{args.env}'
    return args

def get_args_sampleedge():
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_process/deepcad_parsed', 
                        help='Path to data folder')  
    parser.add_argument('--list', type=str, default='data_process/deepcad_data_split_6bit.pkl', 
                        help='Path to data list')  
    parser.add_argument('--surfvae', type=str, default='proj_log/deepcad_surfvae/epoch_400.pt', 
                        help='Path to pretrained surface vae weights')  
    parser.add_argument('--edgevae', type=str, default='proj_log/deepcad_edgevae/epoch_300.pt', 
                        help='Path to pretrained edge vae weights')  
    parser.add_argument("--option", type=str, choices=['ADJ','GEdge','GEvpAC','GEvp','GEdgePosSim','surfpos', 'surfz', 'edgepos', 'edgez','full','GeoEdge','total','GEdgePos','GEdgeZ','PreEdge'], default='total', 
                        help="Choose between option [surfpos,edgepos,surfz,edgez] (default: surfpos)")
    parser.add_argument("--beta_schedule", type=str, choices=['linear','scaled_linear','squaredcos_cap_v2','sigmoid'], default='linear',help="Choose beta schedule type (default: linear)")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')  #512
    parser.add_argument('--train_nepoch', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--test_nepoch', type=int, default=25, help='number of epochs to test model')
    parser.add_argument('--save_nepoch', type=int, default=50, help='number of epochs to save model')
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces') # 50
    parser.add_argument('--max_edge', type=int, default=30, help='maximum number of edges per face') # 30
    parser.add_argument('--threshold', type=float, default=0.05, help='minimum threshold between two faces') #0.05
    parser.add_argument('--bbox_scaled', type=float, default=3, help='scaled the bbox')
    parser.add_argument('--z_scaled', type=float, default=1, help='scaled the latent z')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0, 1], help="GPU IDs to use for training (default: [0, 1])")
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument("--cf",  action='store_true', help='Use data augmentation')
    # Save dirs and reload
    parser.add_argument('--env', type=str, default="surface_pos", help='environment')
    parser.add_argument('--dir_name', type=str, default="proj_log", help='name of the log folder.')
    parser.add_argument("--mode", type=str, choices=['abc', 'deepcad', 'furniture','cadnet'], default='abc', 
                        help="Choose between evaluation mode [abc/deepcad/furniture] (default: abc)")
    parser.add_argument("--model", type=str, choices=['brepgen', 'cadiff'], default='brepgen')
    #args = parser.parse_args()
    args, _ = parser.parse_known_args()
    # saved folder
    args.save_dir = f'{args.dir_name}/{args.env}'
    return args

def rotate_point_cloud(point_cloud, angle_degrees, axis):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Center the point cloud
    center = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - center

    # Apply rotation
    rotated_point_cloud = np.dot(centered_point_cloud, rotation_matrix.T)

    # Translate back to original position
    rotated_point_cloud += center

    # Find the maximum absolute coordinate value
    max_abs_coord = np.max(np.abs(rotated_point_cloud))

    # Scale the point cloud to fit within the -1 to 1 cube
    normalized_point_cloud = rotated_point_cloud / max_abs_coord

    return normalized_point_cloud


def get_bbox(pnts):
    """
    Get the tighest fitting 3D (axis-aligned) bounding box giving a set of points
    """
    bbox_corners = []
    for point_cloud in pnts:
        # Find the minimum and maximum coordinates along each axis
        min_x = np.min(point_cloud[:, 0])
        max_x = np.max(point_cloud[:, 0])

        min_y = np.min(point_cloud[:, 1])
        max_y = np.max(point_cloud[:, 1])

        min_z = np.min(point_cloud[:, 2])
        max_z = np.max(point_cloud[:, 2])

        # Create the 3D bounding box using the min and max values
        min_point = np.array([min_x, min_y, min_z])
        max_point = np.array([max_x, max_y, max_z])
        bbox_corners.append([min_point, max_point])
    return np.array(bbox_corners)


def bbox_corners(bboxes):
    """
    Given the bottom-left and top-right corners of the bbox
    Return all eight corners 
    """
    bboxes_all_corners = []
    for bbox in bboxes:
        bottom_left, top_right = bbox[:3], bbox[3:]
        # Bottom 4 corners
        bottom_front_left = bottom_left
        bottom_front_right = (top_right[0], bottom_left[1], bottom_left[2])
        bottom_back_left = (bottom_left[0], top_right[1], bottom_left[2])
        bottom_back_right = (top_right[0], top_right[1], bottom_left[2])

        # Top 4 corners
        top_front_left = (bottom_left[0], bottom_left[1], top_right[2])
        top_front_right = (top_right[0], bottom_left[1], top_right[2])
        top_back_left = (bottom_left[0], top_right[1], top_right[2])
        top_back_right = top_right

        # Combine all coordinates
        all_corners = [
            bottom_front_left,
            bottom_front_right,
            bottom_back_left,
            bottom_back_right,
            top_front_left,
            top_front_right,
            top_back_left,
            top_back_right,
        ]
        bboxes_all_corners.append(np.vstack(all_corners))
    bboxes_all_corners = np.array(bboxes_all_corners)
    return bboxes_all_corners


def rotate_axis(pnts, angle_degrees, axis, normalized=False):
    """
    Rotate a point cloud around its center by a specified angle in degrees along a specified axis.

    Args:
    - point_cloud: Numpy array of shape (N, ..., 3) representing the point cloud.
    - angle_degrees: Angle of rotation in degrees.
    - axis: Axis of rotation. Can be 'x', 'y', or 'z'.

    Returns:
    - rotated_point_cloud: Numpy array of shape (N, 3) representing the rotated point cloud.
    """

    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)
    
    # Convert points to homogeneous coordinates
    shape = list(np.shape(pnts))
    shape[-1] = 1
    pnts_homogeneous = np.concatenate((pnts, np.ones(shape)), axis=-1)

    # Compute rotation matrix based on the specified axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

    # Apply rotation
    rotated_pnts_homogeneous = np.dot(pnts_homogeneous, rotation_matrix.T)
    rotated_pnts = rotated_pnts_homogeneous[...,:3]

    # Scale the point cloud to fit within the -1 to 1 cube
    if normalized:
        max_abs_coord = np.max(np.abs(rotated_pnts))
        rotated_pnts = rotated_pnts / max_abs_coord

    return rotated_pnts


def rescale_bbox(bboxes, scale):
    # Apply scaling factors to bounding boxes
    scaled_bboxes = bboxes*scale
    return scaled_bboxes


def translate_bbox(bboxes):
    """
    Randomly move object within the cube (x,y,z direction)
    """
    point_cloud = bboxes.reshape(-1,3)
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])
    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])
    x_offset = np.random.uniform( np.min(-1-min_x,0), np.max(1-max_x,0) )
    y_offset = np.random.uniform( np.min(-1-min_y,0), np.max(1-max_y,0) )
    z_offset = np.random.uniform( np.min(-1-min_z,0), np.max(1-max_z,0) )
    random_translation = np.array([x_offset,y_offset,z_offset])
    bboxes_translated = bboxes + random_translation
    return bboxes_translated


def edge2loop(face_edges):
    face_edges_flatten = face_edges.reshape(-1,3)     
    # connect end points by closest distance
    merged_vertex_id = []
    for edge_idx, startend in enumerate(face_edges):
        self_id = [2*edge_idx, 2*edge_idx+1]
        # left endpoint 
        distance = np.linalg.norm(face_edges_flatten - startend[0], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        merged_vertex_id.append(sorted([2*edge_idx, min_id_noself[0]]))
        # right endpoint
        distance = np.linalg.norm(face_edges_flatten - startend[1], axis=1)
        min_id = list(np.argsort(distance))
        min_id_noself = [x for x in min_id if x not in self_id]
        merged_vertex_id.append(sorted([2*edge_idx+1, min_id_noself[0]]))   

    merged_vertex_id = np.unique(np.array(merged_vertex_id),axis=0)
    return merged_vertex_id


def keep_largelist(int_lists):
    # Initialize a list to store the largest integer lists
    largest_int_lists = []

    # Convert each list to a set for efficient comparison
    sets = [set(lst) for lst in int_lists]

    # Iterate through the sets and check if they are subsets of others
    for i, s1 in enumerate(sets):
        is_subset = False
        for j, s2 in enumerate(sets):
            if i!=j and s1.issubset(s2) and s1 != s2:
                is_subset = True
                break
        if not is_subset:
            largest_int_lists.append(list(s1))

    # Initialize a set to keep track of seen tuples
    seen_tuples = set()

    # Initialize a list to store unique integer lists
    unique_int_lists = []

    # Iterate through the input list
    for int_list in largest_int_lists:
        # Convert the list to a tuple for hashing
        int_tuple = tuple(sorted(int_list))
        
        # Check if the tuple is not in the set of seen tuples
        if int_tuple not in seen_tuples:
            # Add the tuple to the set of seen tuples
            seen_tuples.add(int_tuple)
            
            # Add the original list to the list of unique integer lists
            unique_int_lists.append(int_list)
    
    return unique_int_lists

def detect_shared_vertex(edgeV_cad, edge_mask_cad, edgeV_bbox):
    """
    Find the shared vertices 
    """
    edge_id_offset = 2 * np.concatenate([np.array([0]),np.cumsum((edge_mask_cad==False).sum(1))])[:-1]
    valid = True
    
    # Detect shared-vertex on seperate face loop
    used_vertex = []
    face_sep_merges = []
    face_id=0
    for face_idx, (face_edges, face_edges_mask, bbox_edges) in enumerate(zip(edgeV_cad, edge_mask_cad, edgeV_bbox)):
        face_edges = face_edges[~face_edges_mask]
        face_edges = face_edges.reshape(len(face_edges),2,3)
        face_start_id = edge_id_offset[face_idx]  

        # connect end points by closest distance (edge bbox)
        merged_vertex_id = edge2loop(bbox_edges)            
        if len(merged_vertex_id) == len(face_edges):
            merged_vertex_id = face_start_id + merged_vertex_id
            face_sep_merges.append(merged_vertex_id)
            used_vertex.append(bbox_edges*3)
            face_id+=1
            #print('[PASS]')
            continue
        
        # connect end points by closest distance (vertex pos)
        merged_vertex_id = edge2loop(face_edges)
        if len(merged_vertex_id) == len(face_edges):
            merged_vertex_id = face_start_id + merged_vertex_id
            face_sep_merges.append(merged_vertex_id)
            used_vertex.append(face_edges)
            face_id+=1
            #print('[PASS]')
            continue

        print('[FAILED] on '+str(face_id))
        valid = False
        break

    # Invalid
    if not valid:
        assert False

    # Detect shared-vertex across faces 
    total_pnts = np.vstack(used_vertex)
    total_pnts = total_pnts.reshape(len(total_pnts),2,3)
    total_pnts_flatten = total_pnts.reshape(-1,3)

    total_ids = []
    for face_idx, face_merge in enumerate(face_sep_merges):
        # non-self merge centers 
        nonself_face_idx = list(set(np.arange(len(face_sep_merges))) - set([face_idx]))
        nonself_face_merges = [face_sep_merges[x] for x in nonself_face_idx]
        nonself_face_merges = np.vstack(nonself_face_merges)
        nonself_merged_centers = total_pnts_flatten[nonself_face_merges].mean(1)

        # connect end points by closest distance
        across_merge_id = []
        for merge_id in face_merge:
            merged_center = total_pnts_flatten[merge_id].mean(0)
            distance = np.linalg.norm(nonself_merged_centers - merged_center, axis=1)
            nonself_match_id = nonself_face_merges[np.argsort(distance)[0]]
            joint_merge_id = list(nonself_match_id) + list(merge_id)
            across_merge_id.append(joint_merge_id) 
        total_ids += across_merge_id

    # Merge T-junctions
    while (True):
        no_merge = True
        final_merge_id = []

        # iteratelly merge until no changes happen
        for i in range(len(total_ids)):
            perform_merge = False

            for j in range(i+1,len(total_ids)):
                # check if vertex can be further merged 
                max_num = max(len(total_ids[i]), len(total_ids[j]))
                union = set(total_ids[i]).union(set(total_ids[j]))
                common = set(total_ids[i]).intersection(set(total_ids[j]))
                if len(union) > max_num and len(common)>0:
                    final_merge_id.append(list(union))
                    perform_merge = True
                    no_merge = False
                    break

            if not perform_merge:
                final_merge_id.append(total_ids[i]) # no-merge

        total_ids = final_merge_id
        if no_merge: break

    # remove subsets 
    total_ids = keep_largelist(total_ids)

    # merge again base on absolute coordinate value, required for >3 T-junction
    tobe_merged_centers = [total_pnts_flatten[x].mean(0) for x in total_ids]
    tobe_centers = np.array(tobe_merged_centers)
    distances = np.linalg.norm(tobe_centers[:, np.newaxis, :] - tobe_centers, axis=2)
    close_points = distances < 0.1
    mask = np.tril(np.ones_like(close_points, dtype=bool), k=-1)
    non_diagonal_indices = np.where(close_points & mask)
    row_indices, column_indices = non_diagonal_indices

    # update the total_ids
    total_ids_updated = []
    for row, col in zip(row_indices, column_indices):
        total_ids_updated.append(total_ids[row] + total_ids[col])
    for index, ids in enumerate(total_ids):
        if index not in list(row_indices) and index not in list(column_indices):
            total_ids_updated.append(ids)
    total_ids = total_ids_updated

    # merged vertices 
    unique_vertices = []
    for center_id in total_ids:
        center_pnts = total_pnts_flatten[center_id].mean(0) / 3.0
        unique_vertices.append(center_pnts)
    unique_vertices = np.vstack(unique_vertices)

    new_vertex_dict = {}
    for new_id, old_ids in enumerate(total_ids):
        new_vertex_dict[new_id] = old_ids

    return [unique_vertices, new_vertex_dict]

def build_edge_sep_merges(edge_mask_cad):
    """
    根据 edge_mask_cad 构建 edge_sep_merges。

    参数：
        edge_mask_cad (np.ndarray): 面-面邻接矩阵 [num_face, num_face]，
                                    edge_mask_cad[i,j]==False 表示面 i 和面 j 共享边。

    返回：
        edge_sep_merges (list of list of int): 每条共享边对应的两个顶点索引对。
    """
    num_face = edge_mask_cad.shape[0]
    
    # 计算每个面的共享边数量
    shared_edge_counts = (edge_mask_cad == False).sum(axis=1)
    
    # 计算 edge_id_offset
    edge_id_offset = 2 * np.concatenate(([0], np.cumsum(shared_edge_counts)))[:-1]
    
    # 预先计算每个面共享边的面列表
    shared_faces = [np.where(edge_mask_cad[i] == False)[0] for i in range(num_face)]
    
    edge_sep_merges = []
    
    for i in range(num_face):
        # 获取面 i 共享边的所有面 j
        shared_faces_j = shared_faces[i]
        
        for x, j in enumerate(shared_faces_j):
            if j <= i:
                # 只处理上三角部分，避免重复
                continue
            
            # 获取面 j 共享边的所有面
            shared_faces_j_of_j = shared_faces[j]
            
            # 找到面 i 在面 j 的共享边中的位置 y
            y_indices = np.where(shared_faces_j_of_j == i)[0]
            if len(y_indices) == 0:
                raise ValueError(f"面 {j} 没有与面 {i} 共享边的记录。")
            y = y_indices[0]
            
            # 计算顶点索引
            vertex_i_0 = edge_id_offset[i] + 2 * x
            vertex_i_1 = vertex_i_0 + 1
            vertex_j_0 = edge_id_offset[j] + 2 * y
            vertex_j_1 = vertex_j_0 + 1
            
            # 添加两对顶点索引
            edge_sep_merges.append([vertex_i_0, vertex_j_0])
            edge_sep_merges.append([vertex_i_1, vertex_j_1])
    
    return edge_sep_merges

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))  # 初始化每个元素的父节点为自身

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx  # 合并两个集合

def merge_sets_union_find(a_list, b_list):
    n = len(b_list)
    uf = UnionFind(n)  # 初始化并查集

    # 创建元素到集合的映射，使用集合索引
    element_to_sets = defaultdict(list)
    for idx, s in enumerate(b_list):
        for element in s:
            element_to_sets[element].append(idx)

    # 根据 a 中的每个对，合并包含 i 或 j 的所有集合
    for pair in a_list:
        if len(pair) != 2:
            print(f"警告：集合 {pair} 不是一个包含两个元素的集合。跳过。")
            continue  # 跳过不符合要求的集合

        i, j = pair
        sets_i = element_to_sets.get(i, [])
        sets_j = element_to_sets.get(j, [])
        all_related = sets_i + sets_j

        # 合并所有相关集合
        for idx in all_related[1:]:
            uf.union(all_related[0], idx)

    # 聚合集合
    groups = defaultdict(set)
    for idx, s in enumerate(b_list):
        root = uf.find(idx)  # 查找当前集合的根节点
        groups[root].update(s)  # 将当前集合的元素添加到对应的组中

    # 转换为列表
    c = list(groups.values())
    return c


def detect_shared_vertex2(edgeV_cad, edge_mask_cad, edgeV_bbox):
    """
    Find the shared vertices 
    """
    edge_id_offset = 2 * np.concatenate([np.array([0]),np.cumsum((edge_mask_cad==False).sum(1))])[:-1]
    valid = True
    
    # Detect shared-vertex on seperate face loop
    used_vertex = []
    face_sep_merges = []
    merge_num=0
    for face_idx, (face_edges, face_edges_mask, bbox_edges) in enumerate(zip(edgeV_cad, edge_mask_cad, edgeV_bbox)):
        face_edges = face_edges[~face_edges_mask]
        face_edges = face_edges.reshape(len(face_edges),2,3)
        face_start_id = edge_id_offset[face_idx]  

        # connect end points by closest distance (edge bbox)
        merged_vertex_id = edge2loop(bbox_edges)            
        if len(merged_vertex_id) == len(face_edges):
            merged_vertex_id = face_start_id + merged_vertex_id
            face_sep_merges.append(merged_vertex_id)
            used_vertex.append(bbox_edges*3)
            merge_num+=1
            #print('[PASS]')
            continue
        
        # connect end points by closest distance (vertex pos)
        merged_vertex_id = edge2loop(face_edges)
        if len(merged_vertex_id) == len(face_edges):
            merged_vertex_id = face_start_id + merged_vertex_id
            face_sep_merges.append(merged_vertex_id)
            used_vertex.append(face_edges)
            merge_num+=1
            #print('[PASS]')
            continue

        print('[FAILED] on '+str(merge_num))
        valid = False
        break

    # Invalid
    if not valid:
        assert False
    edge_sep_merges=build_edge_sep_merges(edge_mask_cad)
    # Detect shared-vertex across faces 
    total_pnts = np.vstack(used_vertex)
    total_pnts = total_pnts.reshape(len(total_pnts),2,3)
    total_pnts_flatten = total_pnts.reshape(-1,3)
    target_list = []

    # 遍历列表 a
    for arr in face_sep_merges:
        # 对每个 array 转换为包含长度为 2 的集合的列表
        set_list = [set(arr[i]) for i in range(arr.shape[0])]
        # 将转换后的 list 合并到目标列表中
        target_list.extend(set_list)
        
    total_ids=merge_sets_union_find(edge_sep_merges,target_list)

    # remove subsets 
    total_ids = keep_largelist(total_ids)


    # merged vertices 
    unique_vertices = []
    for center_id in total_ids:
        center_pnts = total_pnts_flatten[center_id].mean(0) / 3.0
        unique_vertices.append(center_pnts)
    unique_vertices = np.vstack(unique_vertices)

    new_vertex_dict = {}
    for new_id, old_ids in enumerate(total_ids):
        new_vertex_dict[new_id] = old_ids

    return [unique_vertices, new_vertex_dict]


def detect_shared_edge(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, z_threshold, edge_mask_cad):
    """
    Find the shared edges 
    """
    init_edges = edge_z_cad

    # re-assign edge start/end to unique vertices
    new_ids = []
    for old_id in np.arange(2*len(init_edges)):
        new_id = []
        for key, value in new_vertex_dict.items():
            # Check if the desired number is in the associated list
            if old_id in value:
                new_id.append(key)
        assert len(new_id) == 1 # should only return one unique value
        new_ids.append(new_id[0])

    EdgeVertexAdj = np.array(new_ids).reshape(-1,2)

    # find edges assigned to the same start/end 
    similar_edges = []
    for i, s1 in enumerate(EdgeVertexAdj):
        for j, s2 in enumerate(EdgeVertexAdj):
            if i!=j and set(s1) == set(s2): # same start/end
                z1 = init_edges[i]
                z2 = init_edges[j]
                z_diff = np.abs(z1-z2).mean()
                if z_diff < z_threshold: # check z difference
                    similar_edges.append(sorted([i,j]))
                # else:
                #     print('z latent beyond...')
    similar_edges = np.unique(np.array(similar_edges),axis=0)

    # should reduce total edges by two
    if not 2*len(similar_edges) == len(EdgeVertexAdj):
        assert False, 'edge not reduced by 2'

    # unique edges
    unique_edge_id = similar_edges[:,0]
    EdgeVertexAdj = EdgeVertexAdj[unique_edge_id]
    unique_edges = init_edges[unique_edge_id]

    # unique faces 
    unique_faces = surf_z_cad
    FaceEdgeAdj = []
    ranges = np.concatenate([np.array([0]),np.cumsum((edge_mask_cad==False).sum(1))])
    for index in range(len(ranges)-1):
        adj_ids = np.arange(ranges[index], ranges[index+1])
        new_ids = []
        for id in adj_ids:
            new_id = np.where(similar_edges == id)[0]
            assert len(new_id) == 1
            new_ids.append(new_id[0])
        FaceEdgeAdj.append(new_ids)

    print(f'Post-process: F-{len(unique_faces)} E-{len(unique_edges)} V-{len(unique_vertices)}')

    return [unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj]

def detect_shared_edge2(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, edge_mask_cad):
    """
    Find the shared edges using edge_mask_cad.

    Inputs:
    - unique_vertices: Array of unique vertex positions.
    - new_vertex_dict: Dictionary mapping new vertex IDs to old vertex IDs.
    - edge_z_cad: Array of edge latent vectors (or features).
    - surf_z_cad: Array of face latent vectors (or features).
    - edge_mask_cad: Boolean array indicating edge connections between faces.

    Outputs:
    - unique_faces: Array of unique face latent vectors.
    - unique_edges: Array of unique edge latent vectors.
    - FaceEdgeAdj: List of lists, each sublist contains indices of edges belonging to a face.
    - EdgeVertexAdj: Array of edge-to-vertex adjacency, shape (num_edges, 2).
    """

    # Re-assign edge start/end to unique vertices
    # Compute new edge vertex indices based on new_vertex_dict
    num_edges = len(edge_z_cad)
    new_ids = []

    for old_id in np.arange(2 * num_edges):
        # Find the new vertex ID corresponding to the old vertex ID
        new_id = None
        for key, value in new_vertex_dict.items():
            if old_id in value:
                new_id = key
                break
        assert new_id is not None, f"Old vertex ID {old_id} not found in new_vertex_dict."
        new_ids.append(new_id)

    # Edge-to-vertex adjacency
    EdgeVertexAdj = np.array(new_ids).reshape(-1, 2)

    # Initialize lists to store unique edges and their indices
    unique_edge_tuples = []
    unique_edge_indices = []
    edge_map = {}  # Map from edge tuple to unique edge index

    # Iterate over all edges to identify unique edges using edge_mask_cad
    edge_counter = 0  # Counter for unique edges
    num_faces = edge_mask_cad.shape[0]
    FaceEdgeAdj = []  # List to store edge indices for each face

    # Compute cumulative sum of edges per face to get edge offsets
    edges_per_face = (edge_mask_cad == False).sum(axis=1)
    edge_offsets = np.concatenate([[0], np.cumsum(edges_per_face[:-1])])

    # Build FaceEdgeAdj and identify unique edges
    for face_idx in range(num_faces):
        face_edges = []
        # Get the range of edges for this face
        edge_start = edge_offsets[face_idx]
        edge_end = edge_offsets[face_idx] + edges_per_face[face_idx]
        for edge_idx in range(edge_start, edge_end):
            edge_vertices = tuple(sorted(EdgeVertexAdj[edge_idx]))
            if edge_vertices not in edge_map:
                # This is a unique edge
                edge_map[edge_vertices] = edge_counter
                unique_edge_indices.append(edge_idx)
                unique_edge_tuples.append(edge_vertices)
                face_edges.append(edge_counter)
                edge_counter += 1
            else:
                # Edge already exists, get its index
                face_edges.append(edge_map[edge_vertices])
        FaceEdgeAdj.append(face_edges)
    
    # Build unique_edges array
    unique_edges = edge_z_cad[unique_edge_indices]

    # unique_faces remains the same as surf_z_cad
    unique_faces = surf_z_cad

    num_initial_edges = len(edge_z_cad)
    num_unique_edges = len(unique_edges)

    if not num_unique_edges * 2 == num_initial_edges:
        assert False, f'edge not reduced by 2: initial edges {num_initial_edges}, unique edges {num_unique_edges}'

    # Update EdgeVertexAdj to only include unique edges
    EdgeVertexAdj = np.array(unique_edge_tuples)

    print(f'Post-process: F-{len(unique_faces)} E-{len(unique_edges)} V-{len(unique_vertices)}')

    return unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj
class STModel(nn.Module):
    def __init__(self, num_edge, num_surf):
        super().__init__()
        self.edge_t = nn.Parameter(torch.zeros((num_edge, 3)))
        self.surf_st = nn.Parameter(torch.FloatTensor([1,0,0,0]).unsqueeze(0).repeat(num_surf,1)) 


def get_bbox_minmax(point_cloud):
    # Find the minimum and maximum coordinates along each axis
    min_x = np.min(point_cloud[:, 0])
    max_x = np.max(point_cloud[:, 0])

    min_y = np.min(point_cloud[:, 1])
    max_y = np.max(point_cloud[:, 1])

    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])

    # Create the 3D bounding box using the min and max values
    min_point = np.array([min_x, min_y, min_z])
    max_point = np.array([max_x, max_y, max_z])
    return (min_point, max_point)


def joint_optimize(surf_ncs, edge_ncs, surfPos, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf):
    """
    Jointly optimize the face/edge/vertex based on topology
    """
    loss_func = ChamferDistance()

    model = STModel(num_edge, num_surf)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )

    # Optimize edges (directly compute)
    edge_ncs_se = edge_ncs[:,[0,-1]]
    edge_vertex_se = unique_vertices[EdgeVertexAdj]

    edge_wcs = []
    print('Joint Optimization...')
    for wcs, ncs_se, vertex_se in zip(edge_ncs, edge_ncs_se, edge_vertex_se):
        # scale
        scale_target = np.linalg.norm(vertex_se[0] - vertex_se[1])
        scale_ncs = np.linalg.norm(ncs_se[0] - ncs_se[1])
        edge_scale = scale_target / scale_ncs

        edge_updated = wcs*edge_scale
        edge_se = ncs_se*edge_scale  

        # offset
        offset = (vertex_se - edge_se)
        offset_rev = (vertex_se - edge_se[::-1])

        # swap start / end if necessary 
        offset_error = np.abs(offset[0] - offset[1]).mean()
        offset_rev_error =np.abs(offset_rev[0] - offset_rev[1]).mean()
        if offset_rev_error < offset_error:
            edge_updated = edge_updated[::-1]
            offset = offset_rev
    
        edge_updated = edge_updated + offset.mean(0)[np.newaxis,np.newaxis,:]
        edge_wcs.append(edge_updated)

    edge_wcs = np.vstack(edge_wcs)

    # Replace start/end points with corner, and backprop change along curve
    for index in range(len(edge_wcs)):
        start_vec = edge_vertex_se[index,0] - edge_wcs[index, 0]
        end_vec = edge_vertex_se[index,1] - edge_wcs[index, -1]
        weight = np.tile((np.arange(32)/31)[:,np.newaxis], (1,3))
        weighted_vec = np.tile(start_vec[np.newaxis,:],(32,1))*(1-weight) + np.tile(end_vec,(32,1))*weight
        edge_wcs[index] += weighted_vec            

    # Optimize surfaces 
    face_edges = []
    for adj in FaceEdgeAdj:
        all_pnts = edge_wcs[adj]
        face_edges.append(torch.FloatTensor(all_pnts).cuda())

    # Initialize surface in wcs based on surface pos
    surf_wcs_init = [] 
    bbox_threshold_min = []
    bbox_threshold_max = []   
    for edges_perface, ncs, bbox in zip(face_edges, surf_ncs, surfPos):
        surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
        edges_perface_flat = edges_perface.reshape(-1, 3).detach().cpu().numpy()
        min_point, max_point = get_bbox_minmax(edges_perface_flat)
        edge_center, edge_scale = compute_bbox_center_and_size(min_point, max_point)
        bbox_threshold_min.append(min_point)
        bbox_threshold_max.append(max_point)

        # increase surface size if does not fully cover the wire bbox 
        if surf_scale < edge_scale:
            surf_scale = 1.05*edge_scale
    
        wcs = ncs * (surf_scale/2) + surf_center
        surf_wcs_init.append(wcs)

    surf_wcs_init = np.stack(surf_wcs_init)

    # optimize the surface offset
    surf = torch.FloatTensor(surf_wcs_init).cuda()
    for iters in range(200):
        surf_scale = model.surf_st[:,0].reshape(-1,1,1,1)
        surf_offset = model.surf_st[:,1:].reshape(-1,1,1,3)
        surf_updated = surf + surf_offset 
        
        surf_loss = 0
        for surf_pnt, edge_pnts in zip(surf_updated, face_edges):
            surf_pnt = surf_pnt.reshape(-1,3)
            edge_pnts = edge_pnts.reshape(-1,3).detach()
            surf_loss += loss_func(surf_pnt.unsqueeze(0), edge_pnts.unsqueeze(0), bidirectional=False, reverse=True) 
        surf_loss /= len(surf_updated) 

        optimizer.zero_grad()
        (surf_loss).backward()
        optimizer.step()

        # print(f'Iter {iters} surf:{surf_loss:.5f}') 

    surf_wcs = surf_updated.detach().cpu().numpy()

    return (surf_wcs, edge_wcs)


def face_optimize(surf_ncs, surfPos,filename):
    # Initialize surface in wcs based on surface pos
    surf_wcs_init = [] 
    for ncs, bbox in zip(surf_ncs, surfPos):
        surf_center, surf_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
        wcs = ncs * (surf_scale/2)*1.05 + surf_center
        # wcs = ncs 
        surf_wcs_init.append(wcs)
    if filename is not None:
        with open(filename, 'w') as file:
            for label, array in enumerate(surf_wcs_init):
                assert array.shape == (32, 32, 3), f"Array at position {label} must have shape (32, 32, 3)"
                for i in range(32):
                    for j in range(32):
                        x, y, z = array[i, j]
                        file.write(f'{x:.18e} {y:.18e} {z:.18e} {label}\n')

    return surf_wcs_init


def save_pkl(surf_ncs, surfPos,edge_ncs, edgePos,corner_wcs,edge_mask_tem,save_path):
    data = {
        'surf_wcs':surf_ncs.astype(np.float32),
        'edge_wcs':edge_ncs.astype(np.float32),
        'corner_wcs':corner_wcs.astype(np.float32),
        'faceEdge_adj':edge_mask_tem,
        'surf_bbox_wcs':surfPos.astype(np.float32),
        'edge_bbox_wcs':edgePos.astype(np.float32),
    }

    #save_path = filename.replace(".xyzc", ".pkl")# os.path.join(filename, data['uid']+'.pkl')
    with open(save_path, "wb") as tf:
        pickle.dump(data, tf)

    return save_path



def edge_optimize(edge_ncs, edgePos, edge_mask_tem, filename, focu=True):

    wcs_init = []  # 用于保存转换后的坐标和标签
    for i in range(edge_ncs.shape[0]):  # 遍历每个面
        for j in range(edge_ncs.shape[1]):  # 遍历每个面的边
            if edge_mask_tem[i][j]:  # 检查边是否有效
                ncs = edge_ncs[i][j]  # 获取该边的 32 个点，形状为 [32, 3]
                bbox = edgePos[i][j]  # 获取该边对应的 bbox，形状为 [6]
                face_id = i  # 使用面索引 i 作为 face_id

                # 进行 wcs 坐标的转换
                if focu:
                    edge_center, edge_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
                    wcs = ncs * (edge_scale / 2) + edge_center
                else:
                    wcs = ncs

                wcs_init.append((wcs, face_id))  # 存储每条有效边的转换坐标和对应的面 ID

    # 写入文件
    with open(filename, 'w') as file:
        for wcs, face_id in wcs_init:
            # 对于同一个面的所有有效边，使用相同的 face_id 作为标签
            for point in wcs:
                x, y, z = point
                file.write(f'{x:.18e} {y:.18e} {z:.18e} {face_id}\n')


# def edge_optimize(edge_ncs, edgePos,filename,focu=True):
#     edge_ncs=edge_ncs
#     # Initialize surface in wcs based on surface pos
#     wcs_init = [] 
#     for ncs, bbox in zip(edge_ncs, edgePos):
#         if focu:
#             edge_center, edge_scale = compute_bbox_center_and_size(bbox[0:3], bbox[3:])
#             wcs = ncs * (edge_scale/2) + edge_center
#         else:    
#             wcs = ncs 
#         wcs_init.append(wcs)

#     with open(filename, 'w') as file:
#         for label, array in enumerate(wcs_init):
#             #assert array.shape == (32, 3), f"Array at position {label} must have shape (32, 32, 3)"
#             for i in range(array.shape[0]):
#                     x, y, z = array[i]
#                     file.write(f'{x:.18e} {y:.18e} {z:.18e} {label}\n')

    return wcs_init
def set_seed(seed):
    random.seed(seed)  # Python 的随机数生成器
    np.random.seed(seed)  # NumPy 的随机数生成器
    torch.manual_seed(seed)  # PyTorch 的 CPU 随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch 的 GPU 随机数生成器
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU

    # 保证在多线程中生成相同的随机数
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_mask(batch_size, x, y, even_flag=False):
    if even_flag:
        # 调整 x 和 y，确保它们为偶数
        # 如果 x 是奇数，调整为下一个偶数
        if x % 2 != 0:
            x += 1
        # 如果 y 是奇数，调整为上一个偶数
        if y % 2 != 0:
            y -= 1
        
        # 生成范围在 [x//2, y//2) 的整数，然后乘以 2 保证为偶数
        num = torch.randint(low=x//2, high=y//2, size=(batch_size,)) * 2
    else:
        # 生成范围在 [x, y) 的随机整数
        num = torch.randint(low=x, high=y, size=(batch_size,))
    
    # 生成形状为 (batch_size, y) 的全为 True 的 mask
    mask = torch.ones(batch_size, y, dtype=torch.bool)
    
    # 根据 num 的值设置前 num[i] 个元素为 False
    for i in range(batch_size):
        mask[i, :num[i]] = False
    
    return num, mask
    
    return num, mask
def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)


def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)

        # wire_fixer.SetClosedWireMode(True)
        # wire_fixer.SetFixConnectedMode(True)
        # wire_fixer.SetFixSeamMode(True)

        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()
        # assert ok


def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face

from OCC.Core.BRepCheck import BRepCheck_Analyzer, BRepCheck_NoError
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopExp import TopExp_Explorer
def is_solid_watertight_verbose(solid):
    """
    检查实体是否水密，并打印详细的错误信息。

    参数:
        solid: 要检查的TopoDS_Solid对象。

    返回:
        bool: 如果实体水密且有效，则返回True；否则返回False。
    """
    analyzer = BRepCheck_Analyzer(solid)
    if analyzer.IsValid():
        return True
    else:
        print("实体存在以下问题：")
        shape_map = TopTools_IndexedMapOfShape()
        exp = TopExp_Explorer(solid, 0)  # 0 表示检查所有子形状
        while exp.More():
            sub_shape = exp.Current()
            status = analyzer.Result(sub_shape).Status()
            if status != BRepCheck_NoError:
                print(f"Shape: {sub_shape}, Error status: {status}")
            exp.Next()
        return False
def construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj):
    """
    Fit parametric surfaces / curves and trim into B-rep
    """
    print('Building the B-rep...')
    # Fit surface bspline
    recon_faces = []  
    for points in surf_wcs:
        num_u_points, num_v_points = 32, 32
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1,num_u_points+1):
            for v_index in range(1,num_v_points+1):
                pt = points[u_index-1, v_index-1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face =  GeomAPI_PointsToBSplineSurface(uv_points_array, 3, 8, GeomAbs_C2, 5e-2).Surface() 
        recon_faces.append(approx_face)

    recon_edges = []
    for points in edge_wcs:
        num_u_points = 32
        u_points_array = TColgp_Array1OfPnt(1, num_u_points)
        for u_index in range(1,num_u_points+1):
            pt = points[u_index-1]
            point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            u_points_array.SetValue(u_index, point_2d)
        try:
            approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-3).Curve()  
        except Exception as e:
            print('high precision failed, trying mid precision...')
            try:
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 8e-3).Curve()  
            except Exception as e:
                print('mid precision failed, trying low precision...')
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 0, 8, GeomAbs_C2, 5e-2).Curve()
        recon_edges.append(approx_edge)

    # Create edges from the curve list
    edge_list = []
    for curve in recon_edges:
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        edge_list.append(edge)

    # Cut surface by wire 
    post_faces = []
    post_edges = []
    for idx,(surface, edge_incides) in enumerate(zip(recon_faces, FaceEdgeAdj)):
        corner_indices = EdgeVertexAdj[edge_incides]
        
        # ordered loop
        loops = []
        ordered = [0]
        seen_corners = [corner_indices[0,0], corner_indices[0,1]]
        next_index = corner_indices[0,1]

        while len(ordered)<len(corner_indices):
            while True:
                next_row = [idx for idx, edge in enumerate(corner_indices) if next_index in edge and idx not in ordered]
                if len(next_row) == 0:
                    break
                ordered += next_row
                next_index = list(set(corner_indices[next_row][0]) - set(seen_corners))
                if len(next_index)==0:break
                else: next_index = next_index[0]
                seen_corners += [corner_indices[next_row][0][0], corner_indices[next_row][0][1]]
            
            cur_len = int(np.array([len(x) for x in loops]).sum()) # add to inner / outer loops
            loops.append(ordered[cur_len:])
            
            # Swith to next loop
            next_corner =  list(set(np.arange(len(corner_indices))) - set(ordered))
            if len(next_corner)==0:break
            else: next_corner = next_corner[0]
            next_index = corner_indices[next_corner][0]
            ordered += [next_corner]
            seen_corners += [corner_indices[next_corner][0], corner_indices[next_corner][1]]
            next_index = corner_indices[next_corner][1]

        # Determine the outer loop by bounding box length (?)
        bbox_spans = [get_bbox_norm(edge_wcs[x].reshape(-1,3)) for x in loops]
        
        # Create wire from ordered edges
        _edge_incides_ = [edge_incides[x] for x in ordered]
        edge_post = [edge_list[x] for x in _edge_incides_]
        post_edges += edge_post

        out_idx = np.argmax(np.array(bbox_spans))
        inner_idx = list(set(np.arange(len(loops))) - set([out_idx]))

        # Outer wire
        wire_builder = BRepBuilderAPI_MakeWire()
        for edge_idx in loops[out_idx]:
            wire_builder.Add(edge_list[edge_incides[edge_idx]])
        outer_wire = wire_builder.Wire()

        # Inner wires
        inner_wires = []
        for idx in inner_idx:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_idx in loops[idx]:
                wire_builder.Add(edge_list[edge_incides[edge_idx]])
            inner_wires.append(wire_builder.Wire())
    
        # Cut by wires
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire)
        for wire in inner_wires:
            face_builder.Add(wire)
        face_occ = face_builder.Shape()
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        post_faces.append(face_occ)

    # Sew faces into solid 
    sewing = BRepBuilderAPI_Sewing()
    for face in post_faces:
        sewing.Add(face)
        
    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # Make a solid from the shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()
    if is_solid_watertight_verbose(solid):
        print("实体是水密的。")
    else:
        print("实体不是水密的。")
    return solid



import numpy as np
import trimesh
import os

def build_mesh_from_surf_wcs(surf_wcs, num_u_points=32, num_v_points=32):
    """
    从 surf_wcs 构建三角形网格。
    
    参数：
        surf_wcs (list of np.ndarray): 每个元素是一个形状为 (num_u_points, num_v_points, 3) 的 NumPy 数组，表示一个曲面的 UV 采样点。
        num_u_points (int): u 方向的采样点数。
        num_v_points (int): v 方向的采样点数。
    
    返回：
        trimesh.Trimesh: 合并后的三角形网格。
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0  # 用于调整每个曲面的面索引

    for idx, surface in enumerate(surf_wcs):
        #print(f"正在处理第 {idx + 1} 个曲面...")
        if surface.shape != (num_u_points, num_v_points, 3):
            print(f"警告: surf_wcs[{idx}] 的形状为 {surface.shape}，预期为 ({num_u_points}, {num_v_points}, 3)。跳过此曲面。")
            continue
        
        # 将 (u, v, 3) 的网格展开为 (num_u_points * num_v_points, 3)
        vertices = surface.reshape(-1, 3)
        all_vertices.append(vertices)
        
        # 为每个四边形生成两个三角形
        for u in range(num_u_points - 1):
            for v in range(num_v_points - 1):
                # 当前四边形的四个顶点索引
                idx0 = u * num_v_points + v
                idx1 = (u + 1) * num_v_points + v
                idx2 = (u + 1) * num_v_points + (v + 1)
                idx3 = u * num_v_points + (v + 1)
                
                # 两个三角形
                all_faces.append([idx0 + vertex_offset, idx1 + vertex_offset, idx2 + vertex_offset])
                all_faces.append([idx0 + vertex_offset, idx2 + vertex_offset, idx3 + vertex_offset])
        
        vertex_offset += num_u_points * num_v_points  # 更新顶点偏移量

    if not all_vertices or not all_faces:
        raise ValueError("未成功生成任何顶点或面。请检查 surf_wcs 数据。")

    # 合并所有顶点和面
    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.array(all_faces)

    # 创建 Trimesh 对象
    mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=False)
    
    # 清理和优化网格
    #print("清理和优化网格...")
    #mesh.remove_duplicate_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    # mesh.fill_holes()  # 可选：尝试填补孔洞
    
    #print("网格构建完成。")
    return mesh

def save_mesh_as_stl(mesh, filename):
    """
    将 Trimesh 网格保存为 STL 文件。
    
    参数：
        mesh (trimesh.Trimesh): 要保存的网格。
        filename (str): 输出 STL 文件的路径。
    """
    # 获取目录路径
    directory = os.path.dirname(filename)
    
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            #print(f"目录已创建: {directory}")
        except Exception as e:
            print(f"错误: 无法创建目录 {directory}。详细信息: {e}")
            return
    
    # 尝试导出 STL 文件
    try:
        mesh.export(filename)
        print(f"网格已成功保存为 {filename}")
    except Exception as e:
        print(f"错误: 无法保存 STL 文件。详细信息: {e}")

def construct_and_save_stl(surf_wcs, output_filename='output_model.stl'):
    """
    从 surf_wcs 构建 3D 模型并保存为 STL 文件。
    
    参数：
        surf_wcs (list of np.ndarray): 每个元素是一个形状为 (num_u_points, num_v_points, 3) 的 NumPy 数组，表示一个曲面的 UV 采样点。
        output_filename (str): 输出 STL 文件的路径。
    """
    mesh = build_mesh_from_surf_wcs(surf_wcs)
    save_mesh_as_stl(mesh, output_filename)

def construct_brep_onlysurf(surf_wcs):
    """
    Fit parametric surfaces / curves and trim into B-rep
    """
    print('Building the B-rep...')
    # Fit surface bspline
    recon_faces = []  
    for points in surf_wcs:
        num_u_points, num_v_points = 32, 32
        uv_points_array = TColgp_Array2OfPnt(1, num_u_points, 1, num_v_points)
        for u_index in range(1,num_u_points+1):
            for v_index in range(1,num_v_points+1):
                pt = points[u_index-1, v_index-1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face =  GeomAPI_PointsToBSplineSurface(uv_points_array, 3, 8, GeomAbs_C2, 5e-2).Surface() 
        recon_faces.append(approx_face)

    recon_edges = []
    post_faces = []

    for idx,surface in enumerate(recon_faces):
        # Cut by wires
        face_builder = BRepBuilderAPI_MakeFace(surface)
        face_occ = face_builder.Shape()
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        post_faces.append(face_occ)

    # Sew faces into solid 
    sewing = BRepBuilderAPI_Sewing()
    for face in post_faces:
        sewing.Add(face)
        
    # Perform the sewing operation
    sewing.Perform()
    sewn_shell = sewing.SewedShape()

    # Make a solid from the shell
    maker = BRepBuilderAPI_MakeSolid()
    maker.Add(sewn_shell)
    maker.Build()
    solid = maker.Solid()
    
    return solid
