import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import random
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from diffusionGS.utils.typing import *
from torch.utils.data._utils.collate import default_collate_fn_map
from kiui.cam import orbit_camera, undo_orbit_camera

def read_dnormal(normald_path, cond_pos):
    cond_cam_dis = np.linalg.norm(cond_pos, 2)

    near = 0.867 #sqrt(3) * 0.5
    near_distance = cond_cam_dis - near

    normald = cv2.imread(normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = normald[...,3:]

    depth[depth<near_distance] = 0

    return depth


def _load_single_image(img_path, background_color):
    img = torch.from_numpy(
        np.asarray(
            Image.fromarray(imageio.v2.imread(img_path)).convert("RGBA")
        )
        / 255.0
    ).float()
    mask: Float[Tensor, "H W 1"] = img[:, :, -1:]
    image: Float[Tensor, "H W 3"] = img[:, :, :3] * mask + background_color[
        None, None, :
    ] * (1 - mask)
    return image, mask


@dataclass
class BaseDataModuleConfig:
    local_dir: str = None
    image_dir: str = None
    batch_size: int = 32
    num_workers: int = 0
    num_workers_val: int = 0
    default_fxfy: float = 1422.222 / 1024 
    gen_idxs: Optional[List[int]] = None #top down front left back right
    training_res: Optional[List[int]] = field(default_factory=lambda:[256,256])
    all_idxs: Optional[List[int]] = field(default_factory=lambda:[0,1,2,3,4,5,6,7,8,9,
                                    10,11,12,13,14,15,16,17,18,19,
                                    20,21,22,23,24,27,28,29,
                                    30,31,32,33,34,35,36,37,38,39,]) # hard code for gobjaverse # exclude 25,26, as input

    test_idxs: Optional[List[int]] = field(default_factory=lambda:[0,1,2,3,4,
                                    16,17,18,19])

    gen_rel_idxs: bool=False
    sel_views: int = 4 # first view is the input image
    gen_views: int = 4
    ################################# Image part #################################
    load_image: bool = True
    load_albedo: bool = True
    load_depth: bool = True
    norm_camera: bool = True
    norm_radius: float = 1.8
    background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (1., 1., 1.)
        )


def get_remaining_indices(gen_idx: Optional[List[int]], all_idx: List[int]) -> List[int]:
    """
    从all_idx中去除gen_idx包含的元素，返回剩余元素的列表
    
    参数:
        gen_idx: 要去除的元素列表（可为None，此时返回all_idx的副本）
        all_idx: 原始元素列表
    
    返回:
        去除gen_idx元素后的剩余列表
    """
    # 处理gen_idx为None的情况（返回all_idx的副本）
    if gen_idx is None:
        return all_idx.copy()
    
    # 将gen_idx转换为集合以提高查找效率
    gen_set = set(gen_idx)
    
    # 筛选出all_idx中不在gen_set中的元素
    remaining = [idx for idx in all_idx if idx not in gen_set]
    
    return remaining

class BaseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: BaseDataModuleConfig = cfg
        self.split = split
        self.uids = json.load(open(f'{cfg.local_dir}/{split}.json'))
        self.train_idxs = get_remaining_indices(self.cfg.gen_idxs,self.cfg.all_idxs)
        # Default camera intrinsics [for Gobjaverse]
        self.fxfycxcy = torch.tensor([self.cfg.default_fxfy, self.cfg.default_fxfy, 0.5, 0.5], dtype=torch.float32)
        self.rt_matrix = torch.tensor([
            [1,  0,  0, 0],  # x轴保持不变
            [0,  0,  1, 0],  # y轴 → z轴
            [0,  1,  0, 0],  # z轴 → y轴
            [0,  0,  0, 1]   # 齐次坐标
        ], dtype=torch.float32)

        # load anything
        print(f"Loaded {len(self.uids)} {split} uids")  
    def __len__(self):
        return len(self.uids)

    def _load_png(self, png_bytes: bytes, uint16: bool = False) -> Tensor:
        png = np.frombuffer(png_bytes, np.uint8)
        png = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)  # (H, W, C) ndarray in [0, 255] or [0, 65553]

        png = png.astype(np.float32) / (65535. if uint16 else 255.)  # (H, W, C) in [0, 1]
        png[:, :, :3] = png[:, :, :3][..., ::-1]  # BGR -> RGB
        png_tensor = torch.from_numpy(png).nan_to_num_(0.)  # there are nan in GObjaverse gt normal
        return png_tensor.permute(2, 0, 1)  # (C, H, W) in [0, 1]


    def _load_camera_from_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)  
        # In OpenCV convention
        c2w = np.eye(4)  # float64
        c2w[:3, 0] = np.array(json_dict["x"])
        c2w[:3, 1] = np.array(json_dict["y"])
        c2w[:3, 2] = np.array(json_dict["z"])
        c2w[:3, 3] = np.array(json_dict["origin"])
        return c2w


    def _pick_even_view_indices(self, num_views: int = 4) -> List[int]:
        #borrow from DiffSplat
        assert 12 % num_views == 0  # `12` for even-view sampling in GObjaverse

        if np.random.rand() < 2/3:
            index0 = np.random.choice(range(24))  # 0~23: 24 views in ele from [5, 30]; hard-coded for GObjaverse
            return [(index0 + (24 // num_views)*i) % 24 for i in range(num_views)]
        else:
            index0 = np.random.choice(range(12))  # 27~38: 12 views in ele from [-5, 5]; hard-coded for GObjaverse
            return [((index0 + (12 // num_views)*i) % 12 + 27) for i in range(num_views)]


    def _get_data(self, index):
        uid = self.uids[index]
        ret = {"uid": uid}
        if self.cfg.gen_rel_idxs:
            sel_gen_idxs = self._pick_even_view_indices(self.cfg.gen_views)
        else:
            sel_gen_idxs = self.cfg.gen_idxs

        train_idxs = get_remaining_indices(sel_gen_idxs,self.cfg.all_idxs)
        sel_train_idxs = random.sample(train_idxs, k=self.cfg.sel_views) # hard code for 40 views [GOBJAVERSE]
        all_sel_idxs = sel_gen_idxs + sel_train_idxs
        image_file_dirs = [self.cfg.image_dir + f"{uid}/campos_512_v4/{str(sel_idx).zfill(5)}" for sel_idx in all_sel_idxs]
        # background color
        background_color = torch.as_tensor(self.cfg.background_color)
        rgbs = []
        c2ws = []
        depths = []
        masks = []
        init_azi = None
        for image_file_dir in image_file_dirs:
            #load image
            # breakpoint()
            prefix = os.path.join(image_file_dir,image_file_dir.split('/')[-1])
            img_path = f"{prefix}.png"
            rgb, mask = _load_single_image(img_path, background_color)
            #load camera
            c2w = self._load_camera_from_json(f"{prefix}.json")
            # Blender world + OpenCV cam -> OpenGL world & cam; https://kit.kiui.moe/camera
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1  # invert up and forward direction
            ### if use rel poses ###
            if self.cfg.gen_rel_idxs:
                # Relative azimuth w.r.t. the first view
                ele, azi, dis = undo_orbit_camera(c2w)  # elevation: [-90, 90] from +y(-90) to -y(90)
                if init_azi is None:
                    init_azi = azi
                azi = (azi - init_azi) % 360.  # azimuth: [0, 360] from +z(0) to +x(90)
                # To avoid numerical errors for elevation +/- 90 (GObjaverse index 25 (up) & 26 (down))
                ele_sign = ele >= 0
                ele = abs(ele) - 1e-8
                ele = ele * (1. if ele_sign else -1.)
                new_c2w = torch.from_numpy(orbit_camera(ele, azi, dis)).float()
                c2w = new_c2w
            ##
            normald_path = f"{prefix}_nd.exr"
            depth = read_dnormal(normald_path,c2w[:3, 3:])
            masks.append(torch.tensor(mask))
            depths.append(torch.tensor(depth))
            rgbs.append(rgb)
            c2ws.append(torch.tensor(c2w))
        ret['depths'] = torch.stack(depths).permute(0,3,1,2)
        ret['rgbs'] = torch.stack(rgbs).permute(0,3,1,2)
        ret['masks'] = torch.stack(masks).permute(0,3,1,2)
        ret['c2ws'] = torch.stack(c2ws).to(ret['rgbs'])
        #breakpoint()
        #(torch.norm(ret['c2ws'][0, :3, 3], dim=-1))
        #OpenGL to COLMAP camera for Gaussian renderer
        ret['c2ws'][:, :3, 1:3] *= -1
        # converge to our training setting:（z+ up y- forward）
        ret['c2ws'] = self.rt_matrix @ ret['c2ws'] 
        # breakpoint()
        ## normalize c2w to a fix scale
        # Whether scale the object w.r.t. the first view to a fixed size
        if self.cfg.norm_camera:
            scale = self.cfg.norm_radius / (torch.norm(ret['c2ws'][-1, :3, 3], dim=-1)) #取最后一个相机来作为参考相机
        else:
            scale = 1.
        ret["c2ws"][:, :3, 3] *= scale
        ret['depths'] = ret['depths'] * scale
        # return_dict["cam_pose"][:, 2] *= scale
        ### lift image to training resolutions and fxfycxcy to pixel space ###
        ret['rgbs'] = F.interpolate(ret['rgbs'],size=(self.cfg.training_res[0], self.cfg.training_res[1]))
        ret['depths'] = F.interpolate(ret['depths'],size=(self.cfg.training_res[0], self.cfg.training_res[1]))
        ret['masks'] = F.interpolate(ret['masks'],size=(self.cfg.training_res[0], self.cfg.training_res[1]))
        fxfycxcys = self.fxfycxcy.repeat(ret['rgbs'].shape[0],1)
        fxfycxcys[:,0],fxfycxcys[:,2] = fxfycxcys[:,0]*self.cfg.training_res[0], fxfycxcys[:,2]*self.cfg.training_res[0]
        fxfycxcys[:,1],fxfycxcys[:,3] = fxfycxcys[:,1]*self.cfg.training_res[1], fxfycxcys[:,3]*self.cfg.training_res[1]
        ret['fxfycxcys'] = fxfycxcys
        # first image is input, and the rest images are the image need to gen
        ret['masks_input'] = ret['masks'][:self.cfg.gen_views]
        ret['depths_input'] = ret['depths'][:self.cfg.gen_views]
        ret['rgbs_input'] = ret['rgbs'][:self.cfg.gen_views]
        ret['c2ws_input'] = ret['c2ws'][:self.cfg.gen_views]
        ret['fxfycxcys_input'] = ret['fxfycxcys'][:self.cfg.gen_views]
        return ret
        
    def __getitem__(self, index):
        try:
            return self._get_data(index)
        except Exception as e:
            print(f"Error in {self.uids[index]}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map
        ret = {}
        for key, value in batch[0].items():
            if isinstance(value, str):
                ret[key] = [b[key] for b in batch]
            elif isinstance(value, torch.Tensor):
                ret[key] = torch.stack([b[key] for b in batch])
            elif isinstance(value, np.ndarray):
                ret[key] = torch.stack([torch.from_numpy(b[key]) for b in batch])
            else:
                print(key)
                ret[key] = default_collate_fn_map[type(batch[0][key])](batch)
        return ret