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
# from PIL import Image
import PIL
from diffusionGS.utils.typing import *
from torch.utils.data._utils.collate import default_collate_fn_map


@dataclass
class BaseDataModuleConfig:
    local_dir: str = ''
    local_eval_dir: str = ''
    view_idx_file_path: str = 'extra_files/evaluation_index_re10k.json'
    batch_size: int = 32
    eval_batch_size: int = 1
    eval_subset: int = -1
    num_workers: int = 0
    num_workers_val: int = 0
    training_res: Optional[List[int]] = field(default_factory=lambda:[256,256])


    patch_size: int = 8
    sel_views_train: int = 4
    sel_views: int = 4 # first view is the input image
    scene_scale_factor: float = 1.35
    ################################# Image part #################################
    square_crop: bool = True
    load_image: bool = True


class BaseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: BaseDataModuleConfig = cfg
        self.split = split
        if self.split == 'train':
            with open(cfg.local_dir, 'r') as f:
                all_scene_paths = f.read().splitlines()
            all_scene_paths = [path for path in all_scene_paths if path.strip()]
            self.uids = all_scene_paths#[10:]
        else:
            with open(cfg.local_eval_dir, 'r') as f:
                all_scene_paths = f.read().splitlines()
            all_scene_paths = [path for path in all_scene_paths if path.strip()]
            self.view_idx_list = dict()
            if self.cfg.view_idx_file_path != '':
                if os.path.exists(self.cfg.view_idx_file_path):
                    with open(self.cfg.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        # filter out None values, i.e. scenes that don't have specified input and targetviews
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]
                    filtered_scene_paths = []
                    for scene in all_scene_paths:
                        file_name = scene.split("/")[-1]
                        scene_name = file_name.split(".")[0]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)
                    all_scene_paths = filtered_scene_paths
                    if self.cfg.eval_subset > 0:
                        all_scene_paths = all_scene_paths[:self.cfg.eval_subset]
            self.uids = all_scene_paths
        # load anything
        print(f"Loaded {len(self.uids)} {split} uids")  
        # breakpoint()
    def __len__(self):
        return len(self.uids)

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.cfg.training_res[0]
        patch_size = self.cfg.patch_size
        square_crop = self.cfg.square_crop

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            # if torch.distributed.get_rank() == 0:
            #     import ipdb; ipdb.set_trace()

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws


    def _get_data(self, index):
        uid = self.uids[index]
        scene_path = uid.strip()
        data_json = json.load(open(scene_path, 'r'))
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]
        #### conduct view selecting ####
        if self.split != 'train' and scene_name in self.view_idx_list:
            current_view_idx = self.view_idx_list[scene_name]
            image_indices= current_view_idx["context"][:1] + current_view_idx["target"] # only take first frame as input
            # breakpoint()
        else:
            # breakpoint()
            image_indices = random.sample(range(len(frames)), self.cfg.sel_views + self.cfg.sel_views_train)
        ######### load data #########
        image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
        frames_chosen = [frames[ic] for ic in image_indices]
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)

        # centerize and scale the poses (for unbounded scenes)
        scene_scale_factor = self.cfg.scene_scale_factor
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, index)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]
        # breakpoint()
        ret ={
            'rgbs': input_images,
            'c2ws': input_c2ws,
            'fxfycxcys': input_intrinsics,
            'image_indices': image_indices,
            'scene_indices': scene_indices,
            'masks': torch.ones_like(input_images[:, :1]),
            'indices': indices,
            'uid': scene_name
        }
        # breakpoint()
        ret['rgbs_input'] = ret['rgbs'][:self.cfg.sel_views+1]
        ret['c2ws_input'] = ret['c2ws'][:self.cfg.sel_views+1]
        ret['masks_input'] = ret['masks'][:self.cfg.sel_views+1]
        ret['fxfycxcys_input'] = ret['fxfycxcys'][:self.cfg.sel_views+1]
        # breakpoint()
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