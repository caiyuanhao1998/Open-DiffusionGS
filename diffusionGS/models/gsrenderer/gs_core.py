# conda install -y pytorch  pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install Pillow plyfile opencv-python numpy git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
import math
import os

import cv2
import matplotlib
import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import rearrange
from plyfile import PlyData, PlyElement
from torch import nn

from collections import OrderedDict
import imageio
from typing import Union
from pdb import set_trace as stx


import kiui
from diffusionGS.utils.mesh_utils import decimate_mesh, clean_mesh
import trimesh
def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)


@torch.no_grad()
def get_turntable_cameras(
    hfov=50,
    num_views=8,
    w=384,
    h=384,
    radius=2.7,
    elevation=0,
    up_vector=np.array([0, 0, 1]),
):
    fx = w / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    fxfycxcy = (
        np.array([fx, fy, cx, cy]).reshape(1, 4).repeat(num_views, axis=0)
    )  # [num_views, 4]
    azimuths = np.linspace(0, 360, num_views, endpoint=False)
    elevations = np.ones_like(azimuths) * elevation
    c2ws = []
    for elev, azim in zip(elevations, azimuths):
        elev, azim = np.deg2rad(elev), np.deg2rad(azim)
        z = radius * np.sin(elev)
        base = radius * np.cos(elev)
        x = base * np.cos(azim)
        y = base * np.sin(azim)
        cam_pos = np.array([x, y, z])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        R = np.stack((right, -up, forward), axis=1)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((R, cam_pos[:, None]), axis=1)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)  # [num_views, 4, 4]
    return w, h, num_views, fxfycxcy, c2ws


def imageseq2video(images, filename, fps=24):
    # if images is uint8, convert to float32
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    imageio.mimwrite(filename, images, fps=fps, quality=8)


# copied from: utils.general_utils
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


# copied from: utils.sh_utils
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def create_video(image_folder, output_video_file, framerate=30):
    # Get all image file paths to a list.
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()

    # Read the first image to know the height and width
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        output_video_file, cv2.VideoWriter_fourcc(*"mp4v"), framerate, (width, height)
    )

    # iterate over each image and add it to the video sequence
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


class Camera(nn.Module):
    def __init__(self, C2W, fxfycxcy, h, w):
        """
        C2W: 4x4 camera-to-world matrix; opencv convention
        fxfycxcy: 4
        """
        super().__init__()
        self.C2W = C2W.clone().float()
        self.W2C = self.C2W.inverse()
        self.h = h
        self.w = w

        self.znear = 0.01
        self.zfar = 100.0

        fx, fy, cx, cy = fxfycxcy[0], fxfycxcy[1], fxfycxcy[2], fxfycxcy[3]
        self.tanfovX = w / (2 * fx)
        self.tanfovY = h / (2 * fy)

        def getProjectionMatrix(W, H, fx, fy, cx, cy, znear, zfar):
            P = torch.zeros(4, 4, device=fx.device)
            P[0, 0] = 2 * fx / W
            P[1, 1] = 2 * fy / H
            P[0, 2] = 2 * (cx / W) - 1
            P[1, 2] = 2 * (cy / H) - 1
            P[2, 2] = -(zfar + znear) / (zfar - znear)
            P[3, 2] = 1.0
            P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
            return P

        self.world_view_transform = self.W2C.transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            self.w, self.h, fx, fy, cx, cy, self.znear, self.zfar
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.C2W[:3, 3]


# modified from scene/gaussian_model.py
# 从 Gaussian Splatting 中拿来的代码
class GaussianModel:
    # 将各类函数 setup
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.inv_scaling_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.opacity_activation = torch.sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation

    def __init__(self, sh_degree: int, scaling_modifier=None):
        self.sh_degree = sh_degree
        self._xyz = torch.empty(0)              # 创建一个空的张量
        self._features_dc = torch.empty(0)
        if self.sh_degree > 0:
            self._features_rest = torch.empty(0)
        else:
            self._features_rest = None
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()             # 将各类函数 setup

        self.scaling_modifier = scaling_modifier

    # 重新初始化
    def empty(self):
        self.__init__(self.sh_degree, self.scaling_modifier)

    # 将各个属性设置好
    def set_data(self, xyz, features, scaling, rotation, opacity):
        """
        xyz : torch.tensor of shape (N, 3)
        features : torch.tensor of shape (N, (self.sh_degree + 1) ** 2, 3)
        scaling : torch.tensor of shape (N, 3)
        rotation : torch.tensor of shape (N, 4)
        opacity : torch.tensor of shape (N, 1)
        """
        self._xyz = xyz
        self._features_dc = features[:, :1, :].contiguous()
        if self.sh_degree > 0:
            self._features_rest = features[:, 1:, :].contiguous()
        else:
            self._features_rest = None
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
        return self

    # 放到 cuda 上边
    def to(self, device):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        if self.sh_degree > 0:
            self._features_rest = self._features_rest.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        return self

    # 用一个 mask 去把一些数据选出来，为什么要 filter 掉一些数据呢？
    # 因为要 crop 出中间的一个 Gaussian Splatting
    '''
        主要有以下几个地方需要 filter:
        [1] 点云的位置超出了取值范围
        [2] 近平面和远平面之外的点云
        [3] adaptive density control 时需要 prune 的点云
    '''
    def filter(self, valid_mask):
        self._xyz = self._xyz[valid_mask]
        self._features_dc = self._features_dc[valid_mask]
        if self.sh_degree > 0:
            self._features_rest = self._features_rest[valid_mask]
        self._scaling = self._scaling[valid_mask]
        self._rotation = self._rotation[valid_mask]
        self._opacity = self._opacity[valid_mask]
        return self

    # xyz 的取值范围都限定在 [-1, 1] 之间
    def crop(self, crop_bbx=[-1, 1, -1, 1, -1, 1]):
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bbx
        xyz = self._xyz
        invalid_mask = (
            (xyz[:, 0] < x_min)
            | (xyz[:, 0] > x_max)
            | (xyz[:, 1] < y_min)
            | (xyz[:, 1] > y_max)
            | (xyz[:, 2] < z_min)
            | (xyz[:, 2] > z_max)
        )
        valid_mask = ~invalid_mask

        return self.filter(valid_mask)

    def prune(self, opacity_thres=0.05):
        opacity = self.get_opacity.squeeze(1)
        valid_mask = opacity > opacity_thres

        return self.filter(valid_mask)

    def prune_by_nearfar(self, cam_origins, nearfar_percent=(0.01, 0.99)):
        # cam_origins: [num_cams, 3]
        # nearfar_percent: [near, far]
        assert len(nearfar_percent) == 2
        assert nearfar_percent[0] < nearfar_percent[1]
        assert nearfar_percent[0] >= 0 and nearfar_percent[1] <= 1

        device = self._xyz.device
        # compute distance of all points to all cameras
        # [num_points, num_cams]
        '''
            torch.cdist 用来计算 self._xyz 中的每个点到 cam_origins 中的每个点之间的距离, dists 的 shape 为 [num_points, num_cams]
            [None] 用来增加矩阵的维度
        '''
        dists = torch.cdist(self._xyz[None], cam_origins[None].to(device))[0]


        # [2, num_cams]
        '''
            torch.quantile 用来计算分位数的
            第一行是近平面距离 - 0.01 分位数
            第二行是远平面距离 - 0.99 分位数
        '''
        dists_percentile = torch.quantile(
            dists, torch.tensor(nearfar_percent).to(device), dim=0
        )
        # prune all points that are outside the percentile range
        # [num_points, num_cams]
        # goal: prune points that are too close or too far from any camera
        reject_mask = (dists < dists_percentile[0:1, :]) | (
            dists > dists_percentile[1:2, :]
        )
        reject_mask = reject_mask.any(dim=1)
        valid_mask = ~reject_mask

        return self.filter(valid_mask)

    def apply_all_filters(
        self,
        opacity_thres=0.05,
        crop_bbx=[-1, 1, -1, 1, -1, 1],
        cam_origins=None,
        nearfar_percent=(0.005, 1.0),
    ):
        self.prune(opacity_thres)
        if crop_bbx is not None:
            self.crop(crop_bbx)                                     # cropbox 是针对中心矩阵的
        if cam_origins is not None:
            self.prune_by_nearfar(cam_origins, nearfar_percent)     # nearfar 是针对相机原点的
        return self

    # 将中间的 bbox 缩小
    def shrink_bbx(self, drop_ratio=0.05):
        xyz = self._xyz
        xyz_min, xyz_max = torch.quantile(
            xyz,
            torch.tensor([drop_ratio, 1 - drop_ratio]).float().to(xyz.device),
            dim=0,
        )  # [2, N]
        xyz_min = xyz_min.detach().cpu().numpy()
        xyz_max = xyz_max.detach().cpu().numpy()
        crop_bbx = [
            xyz_min[0],
            xyz_max[0],
            xyz_min[1],
            xyz_max[1],
            xyz_min[2],
            xyz_max[2],
        ]
        print(f"Shrinking bbx to {crop_bbx}")
        return self.crop(crop_bbx)

    def report_stats(self):
        print(
            f"xyz: {self._xyz.shape}, {self._xyz.min().item()}, {self._xyz.max().item()}"
        )
        print(
            f"features_dc: {self._features_dc.shape}, {self._features_dc.min().item()}, {self._features_dc.max().item()}"
        )
        if self.sh_degree > 0:
            print(
                f"features_rest: {self._features_rest.shape}, {self._features_rest.min().item()}, {self._features_rest.max().item()}"
            )
        print(
            f"scaling: {self._scaling.shape}, {self._scaling.min().item()}, {self._scaling.max().item()}"
        )
        print(
            f"rotation: {self._rotation.shape}, {self._rotation.min().item()}, {self._rotation.max().item()}"
        )
        print(
            f"opacity: {self._opacity.shape}, {self._opacity.min().item()}, {self._opacity.max().item()}"
        )

        print(
            f"after activation, xyz: {self.get_xyz.shape}, {self.get_xyz.min().item()}, {self.get_xyz.max().item()}"
        )
        print(
            f"after activation, features: {self.get_features.shape}, {self.get_features.min().item()}, {self.get_features.max().item()}"
        )
        print(
            f"after activation, scaling: {self.get_scaling.shape}, {self.get_scaling.min().item()}, {self.get_scaling.max().item()}"
        )
        print(
            f"after activation, rotation: {self.get_rotation.shape}, {self.get_rotation.min().item()}, {self.get_rotation.max().item()}"
        )
        print(
            f"after activation, opacity: {self.get_opacity.shape}, {self.get_opacity.min().item()}, {self.get_opacity.max().item()}"
        )
        print(
            f"after activation, covariance: {self.get_covariance().shape}, {self.get_covariance().min().item()}, {self.get_covariance().max().item()}"
        )

    '''
        两种装饰器的用法和区别：
        (1) @property, 把类的方法变成属性来调用, 例如: gs.get_xyz, 不需要加括号
        (2) @staticmethod, 静态方法, 不需要实例化, 直接通过类名调用这个类内的方法
    '''

    @property
    def get_scaling(self):
        if self.scaling_modifier is not None:
            return self.scaling_activation(self._scaling) * self.scaling_modifier
        else:
            return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        if self.sh_degree > 0:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            return self._features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    # 设置好各个属性变量的数值类型
    def construct_dtypes(self, use_fp16=False, enable_gs_viewer=True):
        if not use_fp16:
            l = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                l.append((f"f_dc_{i}", "f4"))

            if enable_gs_viewer:
                assert self.sh_degree <= 3, "GS viewer only supports SH up to degree 3"
                sh_degree = 3
                for i in range(((sh_degree + 1) ** 2 - 1) * 3):
                    l.append((f"f_rest_{i}", "f4"))
            else:
                if self.sh_degree > 0:
                    for i in range(
                        self._features_rest.shape[1] * self._features_rest.shape[2]
                    ):
                        l.append((f"f_rest_{i}", "f4"))

            l.append(("opacity", "f4"))
            for i in range(self._scaling.shape[1]):
                l.append((f"scale_{i}", "f4"))
            for i in range(self._rotation.shape[1]):
                l.append((f"rot_{i}", "f4"))
        else:
            l = [
                ("x", "f2"),
                ("y", "f2"),
                ("z", "f2"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                l.append((f"f_dc_{i}", "f2"))

            if self.sh_degree > 0:
                for i in range(
                    self._features_rest.shape[1] * self._features_rest.shape[2]
                ):
                    l.append((f"f_rest_{i}", "f2"))
            l.append(("opacity", "f2"))
            for i in range(self._scaling.shape[1]):
                l.append((f"scale_{i}", "f2"))
            for i in range(self._rotation.shape[1]):
                l.append((f"rot_{i}", "f2"))
        return l


    # dump 点云文件
    def save_ply(
        self,
        path,
        use_fp16=False,
        enable_gs_viewer=True,
        color_code=False,
        filter_mask=None,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        if not color_code:
            rgb = (SH2RGB(f_dc) * 255.0).clip(0.0, 255.0).astype(np.uint8)
        else:
            # use an color map to color code the index of points
            index = np.linspace(0, 1, xyz.shape[0])
            rgb = matplotlib.colormaps["viridis"](index)[..., :3]
            rgb = (rgb * 255.0).clip(0.0, 255.0).astype(np.uint8)

        opacities = self._opacity.detach().cpu().numpy()
        if self.scaling_modifier is not None:
            scale = self.inv_scaling_activation(self.get_scaling).detach().cpu().numpy()
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = self.construct_dtypes(use_fp16, enable_gs_viewer)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        f_rest = None
        if self.sh_degree > 0:
            f_rest = (
                self._features_rest.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )  # (3, (self.sh_degree + 1) ** 2 - 1)

        if enable_gs_viewer:
            sh_degree = 3
            if f_rest is None:
                f_rest = np.zeros(
                    (xyz.shape[0], 3 * ((sh_degree + 1) ** 2 - 1)), dtype=np.float32
                )
            elif f_rest.shape[1] < 3 * ((sh_degree + 1) ** 2 - 1):
                f_rest_pad = np.zeros(
                    (xyz.shape[0], 3 * ((sh_degree + 1) ** 2 - 1)), dtype=np.float32
                )
                f_rest_pad[:, : f_rest.shape[1]] = f_rest
                f_rest = f_rest_pad

        if f_rest is not None:
            attributes = np.concatenate(
                (xyz, rgb, f_dc, f_rest, opacities, scale, rotation), axis=1
            )
        else:
            attributes = np.concatenate(
                (xyz, rgb, f_dc, opacities, scale, rotation), axis=1
            )

        if filter_mask is not None:
            attributes = attributes[filter_mask]
            elements = elements[filter_mask]

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)


    # load 点云文件
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1)
            )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.from_numpy(xyz.astype(np.float32))
        self._features_dc = (
            torch.from_numpy(features_dc.astype(np.float32))
            .transpose(1, 2)
            .contiguous()
        )
        if self.sh_degree > 0:
            self._features_rest = (
                torch.from_numpy(features_extra.astype(np.float32))
                .transpose(1, 2)
                .contiguous()
            )
        self._opacity = torch.from_numpy(
            np.copy(opacities).astype(np.float32)
        ).contiguous()
        self._scaling = torch.from_numpy(scales.astype(np.float32)).contiguous()
        self._rotation = torch.from_numpy(rots.astype(np.float32)).contiguous()

    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        block_size = 2 / num_blocks
        assert resolution % block_size == 0
        split_size = resolution // num_blocks
        ## recenter
        xyzs = self.get_xyz
        opacities = self.get_opacity
        covs = self.get_covariance()
        # normalize to ~ [-1, 1]
        stds = self.get_scaling
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.mesh_center = (mn + mx) / 2
        self.mesh_scale = 1.8 / (mx - mn).amax().item()

        xyzs = (xyzs - self.mesh_center) * self.mesh_scale
        stds = stds * self.mesh_scale

        covs = self.covariance_activation(stds, 1, self._rotation)
        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        val += (mask_opas[:, start:end] * w).sum(-1)
                    
                    # kiui.lo(val, mask_opas, w)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
        
        kiui.lo(occ, verbose=1)
        return occ


    def extract_mesh(self, density_thresh=0.005, resolution=256, decimate_target=1e5):
        occ = self.extract_fields(resolution, num_blocks=64).detach().cpu().numpy()
        import mcubes
        # breakpoint()
        vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1
        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)
        # print(
        #     f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        # )
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        #mesh = Mesh(v=v, f=f, device='cuda')
        return mesh



# 调用 Rasterizer 来进行 render
def render_opencv_cam(
    pc: GaussianModel,
    height: int,
    width: int,
    C2W: torch.Tensor,
    fxfycxcy: torch.Tensor,
    bg_color=(1.0, 1.0, 1.0),
    scaling_modifier=1.0,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.empty_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    )
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    viewpoint_camera = Camera(C2W=C2W, fxfycxcy=fxfycxcy, h=height, w=width)

    bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device)

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.h),
        image_width=int(viewpoint_camera.w),
        tanfovx=viewpoint_camera.tanfovX,
        tanfovy=viewpoint_camera.tanfovY,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }



class DeferredGaussianRender(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        height,
        width,
        C2W,
        fxfycxcy,
        scaling_modifier=None,
        use_gssplat=False,
    ):
        """
        xyz: [b, n_gaussians, 3]
        features: [b, n_gaussians, (sh_degree+1)^2, 3]
        scaling: [b, n_gaussians, 3]
        rotation: [b, n_gaussians, 4]
        opacity: [b, n_gaussians, 1]

        height: int
        width: int
        C2W: [b, v, 4, 4]
        fxfycxcy: [b, v, 4]

        output: [b, v, 3, height, width]
        """
        ctx.scaling_modifier = scaling_modifier

        # Infer sh_degree from features, features 里面存的是 Gaussian 的 attributes
        sh_degree = int(math.sqrt(features.shape[-2])) - 1

        # Create a temp class to hold the data and for rendering
        gaussians_model = GaussianModel(sh_degree, scaling_modifier)

        with torch.no_grad():
            b, v = C2W.size(0), C2W.size(1)
            renders = []
            for i in range(b):
                # 设置高斯的属性
                pc = gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                # 二次循环渲染出每个 view 下的图像
                for j in range(v):
                    renders.append(
                        render_opencv_cam(pc, height, width, C2W[i, j], fxfycxcy[i, j])[
                            "render"
                        ]
                    )
            renders = torch.stack(renders, dim=0)
            renders = renders.reshape(b, v, 3, height, width)

        renders = renders.requires_grad_()

        # Save_for_backward only supports tensors
        ctx.save_for_backward(xyz, features, scaling, rotation, opacity, C2W, fxfycxcy)
        ctx.rendering_size = (height, width)
        ctx.sh_degree = sh_degree
        ctx.use_gssplat = use_gssplat

        # Release the temp class; do not save it.
        del gaussians_model

        return renders

    @staticmethod
    def backward(ctx, grad_output):
        # Restore params
        xyz, features, scaling, rotation, opacity, C2W, fxfycxcy = ctx.saved_tensors
        height, width = ctx.rendering_size
        sh_degree = ctx.sh_degree
        use_gssplat = ctx.use_gssplat

        # **The order of this dict should not be changed**
        input_dict = OrderedDict(
            [
                ("xyz", xyz),
                ("features", features),
                ("scaling", scaling),
                ("rotation", rotation),
                ("opacity", opacity),
            ]
        )
        input_dict = {k: v.detach().requires_grad_() for k, v in input_dict.items()}

        # Create a temp class to hold the data and for rendering
        gaussians_model = GaussianModel(sh_degree, ctx.scaling_modifier)

        with torch.enable_grad():
            b, v = C2W.size(0), C2W.size(1)
            for i in range(b):
                for j in range(v):
                    # The backward will remove the diff graph, thus each time we need a copy
                    pc = gaussians_model.set_data(
                        **{k: v[i] for k, v in input_dict.items()}
                    )

                    # Forward
                    render = render_opencv_cam(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j]
                    )["render"]

                    # Backward, suppose that only values in input_dict will get gradients.
                    render.backward(grad_output[i, j])

        del gaussians_model

        return *[var.grad for var in input_dict.values()], None, None, None, None, None, None


# Function for the class
deferred_gaussian_render = DeferredGaussianRender.apply


class DeferredGaussianRender_scene(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        height,
        width,
        C2W,
        fxfycxcy,
        scaling_modifier=None,
    ):
        """
        xyz: [b, n_gaussians, 3]
        features: [b, n_gaussians, (sh_degree+1)^2, 3]
        scaling: [b, n_gaussians, 3]
        rotation: [b, n_gaussians, 4]
        opacity: [b, n_gaussians, 1]

        height: int
        width: int
        C2W: [b, v, 4, 4]
        fxfycxcy: [b, v, 4]

        output: [b, v, 3, height, width]
        """
        ctx.scaling_modifier = scaling_modifier

        # Infer sh_degree from features, features 里面存的是 Gaussian 的 attributes
        sh_degree = int(math.sqrt(features.shape[-2])) - 1

        # Create a temp class to hold the data and for rendering
        gaussians_model = GaussianModel(sh_degree, scaling_modifier)

        with torch.no_grad():
            b, v = C2W.size(0), C2W.size(1)
            renders = torch.zeros(
                b, v, 3, height, width, dtype=torch.float32, device=xyz.device
            )

            depth = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )
            alpha = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )

            for i in range(b):
                # 设置高斯的属性
                pc = gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                # 二次循环渲染出每个 view 下的图像
                for j in range(v):
                    buffers = render_opencv_cam(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j]
                    )
                    renders[i, j] = buffers["render"]
                    if "depth" in buffers and buffers["depth"] is not None: 
                        depth[i, j] = buffers["depth"]
                    if "alpha" in buffers and buffers["alpha"] is not None:
                        alpha[i, j] = buffers["alpha"]

        renders = renders.requires_grad_()
        # depth = depth.requires_grad_()
        # alpha = alpha.requires_grad_()

        # Save_for_backward only supports tensors
        ctx.save_for_backward(xyz, features, scaling, rotation, opacity, C2W, fxfycxcy)
        ctx.rendering_size = (height, width)
        ctx.sh_degree = sh_degree

        # Release the temp class; do not save it.
        del gaussians_model

        return renders, depth, alpha

    @staticmethod
    def backward(ctx, grad_render, grad_depth, grad_alpha):
        # Restore params
        xyz, features, scaling, rotation, opacity, C2W, fxfycxcy = ctx.saved_tensors
        height, width = ctx.rendering_size
        sh_degree = ctx.sh_degree

        # **The order of this dict should not be changed**
        input_dict = OrderedDict(
            [
                ("xyz", xyz),
                ("features", features),
                ("scaling", scaling),
                ("rotation", rotation),
                ("opacity", opacity),
            ]
        )
        input_dict = {k: v.detach().requires_grad_() for k, v in input_dict.items()}

        # Create a temp class to hold the data and for rendering
        gaussians_model = GaussianModel(sh_degree, ctx.scaling_modifier)

        with torch.enable_grad():
            b, v = C2W.size(0), C2W.size(1)
            for i in range(b):
                for j in range(v):
                    # The backward will remove the diff graph, thus each time we need a copy
                    pc = gaussians_model.set_data(
                        **{k: v[i] for k, v in input_dict.items()}
                    )

                    # Forward
                    buffers = render_opencv_cam(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j]
                    )
                    render = buffers["render"]
                    # if "depth" in buffers and buffers["depth"] is not None: 
                    #     depth = buffers["depth"]
                    # if "alpha" in buffers and buffers["alpha"] is not None:
                    #     alpha = buffers["alpha"]

                    # Backward, suppose that only values in input_dict will get gradients.
                    render.backward(grad_render[i, j])
                    # depth.backward(grad_depth[i, j])
                    # alpha.backward(grad_alpha[i, j])

        del gaussians_model

        return *[var.grad for var in input_dict.values()], None, None, None, None, None

# Function for the class
deferred_gaussian_render_scene = DeferredGaussianRender_scene.apply


@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def render_turntable(pc: GaussianModel, rendering_resolution=384, num_views=8):             # 渲染一圈图像 - 8 个 view
    w, h, v, fxfycxcy, c2w = get_turntable_cameras(
        h=rendering_resolution, w=rendering_resolution, num_views=num_views
    )       # 这里的 v - viewpoint 被重置了，从 get_turntable_cameras 中抽取

    device = pc._xyz.device
    fxfycxcy = torch.from_numpy(fxfycxcy).float().to(device)  # [v, 4]
    c2w = torch.from_numpy(c2w).float().to(device)  # [v, 4, 4]

    renderings = torch.zeros(v, 3, h, w, dtype=torch.float32, device=device)
    for j in range(v):
        renderings[j] = render_opencv_cam(pc, h, w, c2w[j], fxfycxcy[j])["render"]
    torch.cuda.empty_cache()  # free up memory on GPU
    renderings = renderings.detach().cpu().numpy()
    renderings = (renderings * 255).clip(0, 255).astype(np.uint8)
    renderings = rearrange(renderings, "v c h w -> h (v w) c")
    return renderings


class GaussianRenderer(nn.Module):
    def __init__(self, sh_degree, scaling_modifier=None):
        super().__init__()

        self.sh_degree = sh_degree
        self.scaling_modifier = scaling_modifier

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        height,
        width,
        C2W,
        fxfycxcy,
        deferred=True,
    ):
        """
        xyz: [b, n_gaussians, 3]
        features: [b, n_gaussians, (sh_degree+1)^2, 3]
        scaling: [b, n_gaussians, 3]
        rotation: [b, n_gaussians, 4]
        opacity: [b, n_gaussians, 1]

        height: int
        width: int
        C2W: [b, v, 4, 4]
        fxfycxcy: [b, v, 4]

        output: [b, v, 3, height, width]
        """

        if deferred:
            renderings = deferred_gaussian_render(
                xyz,
                features,
                scaling,
                rotation,
                opacity,
                height,
                width,
                C2W,
                fxfycxcy,
                self.scaling_modifier,
            )

            b, v = C2W.size(0), C2W.size(1)
            gaussians_models = []
            for i in range(b):
                pc = GaussianModel(self.sh_degree, self.scaling_modifier)
                pc = pc.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                gaussians_models.append(pc)
        else:
            b, v = C2W.size(0), C2W.size(1)
            renderings = torch.zeros(
                b, v, 3, height, width, dtype=torch.float32, device=xyz.device
            )

            gaussians_models = []
            for i in range(b):
                pc = GaussianModel(self.sh_degree, self.scaling_modifier)
                pc = pc.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                gaussians_models.append(pc)
                for j in range(v):
                    renderings[i, j] = render_opencv_cam(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j]
                    )["render"]
        return renderings, gaussians_models


@torch.no_grad()
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def render_generic(pc: GaussianModel, c2ws, fxfycxcy,  h=512, w=512):
    # cw2s: [v, 4, 4]
    # fxfycxcy: [v, 4]

    v = c2ws.shape[0]
    device = pc._xyz.device
    fxfycxcy = fxfycxcy.float().to(device)  # [v, 4]
    renderings = torch.zeros(v, 3, h, w, dtype=torch.float32, device=device)
    for j in range(v):
        renderings[j] = render_opencv_cam(pc, h, w, c2ws[j], fxfycxcy[j])["render"]
    torch.cuda.empty_cache()  # free up memory on GPU
    renderings = renderings.detach().cpu().numpy()
    renderings = (renderings * 255).clip(0, 255).astype(np.uint8)
    renderings = rearrange(renderings, "v c h w -> v h w c")
    return renderings


if __name__ == "__main__":
    import json

    from PIL import Image
    from tqdm import tqdm

    out_dir = "/mnt/localssd/debug-3dgs"
    os.makedirs(out_dir, exist_ok=True)

    os.system(
        f"wget https://phidias.s3.us-west-2.amazonaws.com/kaiz/neural-capture/eval-3dgs-lowres/AWS_test_set/results/1.fashion_boots_rubber_boots__short__Feb_21__2023_at_5_19_25_PM_yf/point_cloud/iteration_30000_fg/point_cloud.ply -O {out_dir}/point_cloud.ply"
    )
    os.system(
        f"wget https://neural-capture.s3.us-west-2.amazonaws.com/data/AWS_test_set/preprocessed/1.fashion_boots_rubber_boots__short__Feb_21__2023_at_5_19_25_PM_yf/opencv_cameras_traj_norm.json -O {out_dir}/opencv_cameras_traj_norm.json"
    )

    device = "cuda:0"

    pc = GaussianModel(sh_degree=3)
    pc.load_ply(f"{out_dir}/point_cloud.ply")
    pc = pc.to(device)

    # pc.save_ply(f"{out_dir}/point_cloud_shrink.ply")
    # pc.load_ply(f"{out_dir}/point_cloud_shrink.ply")
    # pc = pc.to(device)

    # pc.prune(opacity_thres=0.05)
    # pc.save_ply(f"{out_dir}/point_cloud_shrink_prune.ply")
    # pc = pc.to(device)

    # pc.shrink_bbx(drop_ratio=0.01)
    # pc.save_ply(f"{out_dir}/point_cloud_shrink_prune.ply")
    # pc = pc.to(device)

    pc.report_stats()

    with open(f"{out_dir}/opencv_cameras_traj_norm.json", "r") as f:
        cam_traj = json.load(f)

    for i, cam in tqdm(enumerate(cam_traj["frames"]), desc="Rendering progress"):
        w2c = np.array(cam["w2c"])
        c2w = np.linalg.inv(w2c)
        c2w = torch.from_numpy(c2w.astype(np.float32)).to(device)

        fx = cam["fx"]
        fy = cam["fy"]
        cx = cam["cx"]
        cy = cam["cy"]
        cx = cx - 5
        cy = cy + 4
        fxfycxcy = torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device=device)

        h = cam["h"]
        w = cam["w"]

        im = render_opencv_cam(pc, h, w, c2w, fxfycxcy, bg_color=[0.0, 0.0, 0.0])[
            "render"
        ]
        im = im.detach().cpu().numpy().transpose(1, 2, 0)
        im = (im * 255).astype(np.uint8)
        Image.fromarray(im).save(f"{out_dir}/render_{i:08d}.png")

    create_video(out_dir, f"{out_dir}/render.mp4", framerate=30)
    print(f"Saved {out_dir}/render.mp4")