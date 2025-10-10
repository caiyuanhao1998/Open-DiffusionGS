from dataclasses import dataclass, field

import numpy as np
import json
import copy
import torch
import torch.nn.functional as F
from skimage import measure
from einops import repeat
from tqdm import tqdm
from PIL import Image
import os
import cv2

import diffusionGS
from easydict import EasyDict as edict
from einops import repeat, rearrange
from diffusionGS.utils.misc import get_rank
from diffusionGS.systems.base import BaseSystem
from diffusionGS.models.diffusion import create_diffusion
from diffusionGS.utils.typing import *
from diffusionGS.utils.losses import LossComputer
from diffusionGS.systems.utils import *
#from diffusionGS.models.denoiser.denoiser_utils import TransformInput
import random
# DEBUG = True
@diffusionGS.register("diffusion-gs-system")
class PointDiffusionSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        num_inference_steps: int = 50
        # shape vae model
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)
        # noise scheduler
        noise_scheduler_type: str = None
        noise_scheduler: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = diffusionGS.find(self.cfg.shape_model_type)(self.cfg.shape_model)
        self.noise_scheduler = diffusionGS.find(self.cfg.noise_scheduler_type)(**self.cfg.noise_scheduler)
        self.diffusion_training = create_diffusion(str(1000), predict_xstart=True)
        self.diffusion_inference = create_diffusion(str(self.cfg.num_inference_steps), predict_xstart=True)
        self.loss_computer = LossComputer()
    
    def get_example_data(self):
        example_data_path = "examp_data/example/debug_objaverse_dataset/data_examples/batch_objaverse_example.json"
        with open(example_data_path, 'r') as file:
            batch = json.load(file)
        batch['c2w'] = torch.tensor(np.stack(batch['c2w']))[:4].float().unsqueeze(0)
        batch['fxfycxcy'] = torch.tensor(np.stack(batch['fxfycxcy']))[:4].float().unsqueeze(0)
        batch['image'] = torch.tensor(np.stack(batch['image']))[:4].float().unsqueeze(0)
        batch['index'] = torch.tensor(np.stack(batch['index']))[:4].float().unsqueeze(0)
        batch['depth'] = torch.tensor(np.stack(batch['depth']))[:4].float().unsqueeze(0)
        batch['normal'] = torch.tensor(np.stack(batch['normal']))[:4].float().unsqueeze(0)
        batch = {k: v.cuda() for k, v in batch.items() if k!="camera_json_path" and k!="view_choices"}
        return batch #torch.tensor(images,dtype=torch.float32).unsqueeze(0).to('cuda'), torch.tensor(c2ws,dtype=torch.float32).unsqueeze(0).to('cuda'), torch.tensor(fxfycxcys,dtype=torch.float32).unsqueeze(0).to('cuda')


    def select_train_views(self,images,c2ws,fxfycxcys):
        sel_index = np.random.choice(images.shape[1], 4,replace=False)
        sel_c2ws = c2ws[:,sel_index]
        sel_images = images[:,sel_index]
        sel_fxfycxcys = fxfycxcys[:,sel_index]
        return sel_images, sel_c2ws, sel_fxfycxcys

    def forward(self, batch: Dict[str, Any], skip_noise=False) -> Dict[str, Any]:
        #get training input
        sel_images = batch['rgbs_input'] #(batch['rgbs_input']*2.) - 1. ##归一化
        ray_o, ray_d = TransformInput(sel_images, batch['c2ws_input'], batch['fxfycxcys_input'])
        bs, v, c, h, w = sel_images.shape
        # 3. sample noise that we"ll add to the latents
        noise = torch.randn_like(sel_images)#.to(sel_images)
        # 4. Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.cfg.noise_scheduler.num_train_timesteps,
            (bs,),
            device=sel_images.device,
        )
        timesteps = timesteps.long()
        # breakpoint()
        sel_images[:,1:] = self.diffusion_training.q_sample(sel_images[:,1:], timesteps, noise=noise[:,1:])
        gt_img_aligned_xyz = ray_o + ray_d * batch['depths_input']
        #breakpoint()
        guassians_parameters, img_aligned_xyz = self.shape_model.image_to_gaussians(sel_images, ray_o, ray_d, timesteps)
        ### 5. render image from gen gaussians
        rendered_images = self.shape_model.render_gaussians(guassians_parameters, batch['c2ws'], batch['fxfycxcys'], sel_images.shape[3],sel_images.shape[4])
        #rendered_images,rendered_depths = rendered_images[:,:,:3],rendered_images[:,:,3:]
        l2_loss, lpips_loss, ssim_loss, pointsdist_loss, l2_loss_xyz = self.loss_computer(
                                                                        rendered_images,
                                                                        batch['rgbs'],
                                                                        batch['masks'],
                                                                        batch['masks_input'],
                                                                        ray_o,
                                                                        img_aligned_xyz=img_aligned_xyz,
                                                                        gt_img_aligned_xyz=gt_img_aligned_xyz,
                                                                        )
        # breakpoint()
        return {
            "loss_diffusion": l2_loss.mean(),
            "loss_lpips": lpips_loss.mean(),
            "loss_ssim": ssim_loss.mean(),
            "loss_xyz": l2_loss_xyz.mean(),
            "loss_pointsdist": pointsdist_loss.mean(),
            "latents": sel_images,
            "x_t": sel_images,
            "noise": noise,
            "noise_pred": rendered_images,
            "timesteps": timesteps,
            }


    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_"):
                self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        # TODO: write a save and denoise process
        # out = self(batch)      
            # if get_rank() == 0:
            ######################## for debug ########################
        # input_batch_test = self.get_example_data()
        # ray_o, ray_d = TransformInput(input_batch_test['image'], input_batch_test['c2w'], input_batch_test['fxfycxcy'])
        # input_batch = edict(
        #     image = input_batch_test['image'][:,:1],
        #     c2w = input_batch_test['c2w'],
        #     fxfycxcy = input_batch_test['fxfycxcy'],
        #     ray_o=ray_o,
        #     ray_d=ray_d,
        #     )
        # input_images = input_batch_test['image'] #batch['rgbs_input']
        # image_condition = input_batch_test['image'][0,:1]
            ######################## for debug ########################
        ######################## for real run ########################     
        ray_o, ray_d = TransformInput(batch['rgbs_input'], batch['c2ws_input'], batch['fxfycxcys_input'])
        input_batch = edict(
            image = batch['rgbs_input'][:,:1],
            c2w = batch['c2ws_input'],
            fxfycxcy = batch['fxfycxcys_input'],
            ray_o=ray_o,
            ray_d=ray_d,
            )
        input_images = batch['rgbs_input']  #[:,:1]
        b, v, c, h, w = input_images.shape      # b = n
        sample_noise = torch.randn(b, v-1, c, h, w, device=self.device)  # 在 timestep 为 T 的时候采的 noise
        input_batch["image_noisy"] = sample_noise
        ####################### for real run ########################
        image_condition = input_images[:, 0, ...].unsqueeze(1)
        traj_timesteps = list(self.diffusion_inference.timestep_map)[::-1]         # 提前设计好的 timestep 路线
        traj_len = len(traj_timesteps)

        traj_samples = []
        traj_pred_xstart = []
        final_out = None

        for out in self.diffusion_inference.p_sample_loop_progressive(
            self.shape_model,
            sample_noise.shape,
            input_batch,
            clip_denoised=False,
            progress=True,
            device=self.device,
        ):
            samples = out["sample"]
            pred_xstart = out["pred_xstart"]
            # 把 image condition 和 samples concate 起来
            samples_present = torch.cat((image_condition, samples), dim=1)
            pred_xstart_present = torch.cat((image_condition, pred_xstart), dim=1)
            traj_samples.append(samples_present)
            traj_pred_xstart.append(pred_xstart_present)
            final_out = out

        traj_samples = torch.stack(traj_samples, 0)  # [T, B, V, C, H, W]                   [250, 8, 4, 3, 256, 256]
        traj_pred_xstart = torch.stack(traj_pred_xstart, 0)  # [T, B, V, C, H, W]           [250, 8, 4, 3, 256, 256]

        traj_samples = rearrange(
            traj_samples, "t n v c h w -> n t h (v w) c"
        ).contiguous().cpu().numpy()

        traj_pred_xstart = rearrange(
            traj_pred_xstart, "t n v c h w -> n t h (v w) c"
        ).contiguous().cpu().numpy()

        # save multiple videos
        for i in range(traj_samples.shape[0]):
            vis = traj_samples[i]
            vis = display_timestep_on_video(vis, traj_timesteps)
            self.save_videos(f"it{self.true_global_step}/{batch['uid'][0]}_{batch['sel_idx'][0] if 'sel_idx' in batch.keys() else 0}_{i:04d}_traj_xt.mp4", vis, fps=24, quality=8)

        # 将输出的图片保存为视频
        image_grid = []
        for i in range(traj_pred_xstart.shape[0]):
            vis = traj_pred_xstart[i]
            vis = display_timestep_on_video(vis, traj_timesteps)
            self.save_videos(f"it{self.true_global_step}/{batch['uid'][0]}_{batch['sel_idx'][0] if 'sel_idx' in batch.keys() else 0}_{i:04d}_traj_xstart.mp4", vis, fps=24, quality=8)
        self.save_guassians_ply(f"it{self.true_global_step}/{batch['uid'][0]}_{batch['sel_idx'][0] if 'sel_idx' in batch.keys() else 0}.ply", final_out['denoiser_output_dict']['pred_gaussians'][0],render_video=True)
        self.save_torch_images(f"it{self.true_global_step}/{batch['uid'][0]}_{batch['sel_idx'][0] if 'sel_idx' in batch.keys() else 0}.png", torch.cat([image_condition[0],final_out['denoiser_output_dict']['render_images'][0]], dim=0))
        return {"val/loss": 123}

    def on_validation_epoch_end(self):
        pass