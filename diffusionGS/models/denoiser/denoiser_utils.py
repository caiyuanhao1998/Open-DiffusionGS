import random

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from kornia.geometry.transform import pyrdown
from rich import print
import itertools
import torch.nn.functional as F
from pdb import set_trace as stx


# 将输入的 data_batch 拆分成 input 和 target
class SplitData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self, data_batch, target_has_input=True):
        """
            image (torch.tensor): [b, v, c, h, w]
            fxfycxcy (torch.tensor): [b, v, 4]
            c2w (torch.tensor): [b, v, 4, 4]
            label (torch.tensor): [b, v, c, h, w]
        input:
            data_batch: {"image": (b, v, c, h, w)} - a dict
            举个例子: [8, 8, 3, 256, 256] 
        return:
            input: {"image": (b, v, c, h, w)} - a dict
            target: {"image": (b, v, c, h, w)} - a dict
        """
        input, target = {}, {}

        b,v,c,h,w = data_batch['image'].size()

        index = None
        for key, value in data_batch.items():
            try:
                input[key] = value[:, : self.config.training.num_input_views, ...]      # 取前 num_input_views 个视图, 在应用中 num_input_views = 4
            except:
                breakpoint()
            if self.config.training.num_target_views >= value.size(1):  # 如果 target 的数量大于等于输入的数量, 4 < 8, 这一步不执行
                target[key] = value
            else:
                if index is None:               # 只在 index 尚未被分配时执行
                    b, v = value.shape[:2]
                    '''
                        分两种情况：
                        [1] target 里面包含 input
                        [2] target 里面不包含 input
                    '''
                    if target_has_input:
                        index = np.array(           # 生成一个随机的 index，用于选择 target, input 和 target 可能有重叠
                            [
                                random.sample(
                                    range(v), self.config.training.num_target_views
                                )
                                for _ in range(b)
                            ]
                        )
                    else:
                        assert (
                            self.config.training.num_input_views
                            + self.config.training.num_target_views
                            <= self.config.training.num_views
                        ), "num_input_views + num_target_views must <= num_views to avoid duplicate views"
                        index = np.array(
                            [
                                [
                                    self.config.training.num_views - 1 - j
                                    for j in range(
                                        self.config.training.num_target_views
                                    )
                                ]
                                for _ in range(b)
                            ]
                        )

                    index = torch.from_numpy(index).long().to(value.device)     # 把随机取好的序号放到 GPU 上

                # match shape of index to v, insert dummy dimensions to match v
                value_index = index
                if value.dim() > 2: # 此时包含了维度 v 
                    dummy_dims = [
                        1,
                    ] * (value.dim() - 2)
                    value_index = index.reshape(
                        index.size(0), index.size(1), *dummy_dims
                    )
                try:
                    '''
                        torch.gather - 按照指定的张量在特定的维度按照索引来取元素
                        此处是根据 index 来取 target 的元素
                        在 label wild 测试中, not eval, value: torch.Size([16, 8, 1, 256, 256]),  index: torch.Size([16, 6, 1, 1, 1])
                        在 label wild 测试中, eval, value: torch.Size([16, 8, 1, 256, 256]),  index: torch.Size([16, 6, 1, 1, 1])
                    '''
                    # if key == 'label':
                    #     stx()
                    target[key] = torch.gather(
                        value,
                        dim=1,
                        index=value_index.expand(-1, -1, *value.size()[2:]),
                    )
                except Exception as e:
                    print(f"key: {key}")
                    print(f"value: {value.size()}")
                    print(f"index: {value_index.size()}")
                    raise e
        return edict(input), edict(target)



'''
    [1] 从输入的数据批次中提取图像、内参、外参和索引。
    [2] 对图像、内参和外参的维度进行检查，确保它们的维度符合预期。
    [3] 对图像、内参和外参进行重塑，以便于后续的处理。
    [4] 生成图像的网格坐标，并将这些坐标转移到图像的设备上。
    [5] 对网格坐标进行归一化处理，使其范围在 [-1, 1] 之间。
    [6] 对归一化后的坐标进行扩展，以便于后续的处理。
    [7] 计算射线的方向(ray_d)和原点(ray_o)。
    [8] 如果提供了 patch_size,则会对图像进行分块处理,并计算每个块的颜色、射线的方向和原点。
    [9] 最后，将处理后的数据打包并返回。
'''
class TransformInput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self, data_batch, patch_size=None):
        """transform input image before feeding into transformer

        Args:
            image (torch.tensor): [b, v, c, h, w]               # 图像
            fxfycxcy (torch.tensor): [b, v, 4]                  # 内参
            c2w (torch.tensor): [b, v, 4, 4]                    # 外参

        Returns:
            image: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
        
        Input 这个字典所包含的 key 有: (['image', 'ray_o', 'ray_d', 'ray_d_cam', 'fxfycxcy', 'c2w', 'index', 'xy_norm', 'ray_color_patch', 'ray_o_patch', 'ray_d_patch', 'ray_xy_norm_patch', 'proj_mat'])
        """
        image, fxfycxcy, c2w, image_noisy = (
            data_batch.image,
            data_batch.fxfycxcy,
            data_batch.c2w,
            # data_batch.index,
            data_batch.image_noisy,
        )

        # stx()
        #image[:,1:4,...] = image_noisy      # [8, 4, 3, 256, 256]

        # stx()

        assert image.dim() == 5, f"image dim should be 5, but got {image.dim()}"        # b, v, c, h, w
        assert (
            fxfycxcy.dim() == 3
        ), f"fxfycxcy dim should be 3, but got {fxfycxcy.dim()}"                # b, v, 4
        assert c2w.dim() == 4, f"c2w dim should be 4, but got {c2w.dim()}"      # b, v, 4, 4

        b, v, c, h, w = image.size()
        image = image.reshape(b * v, c, h * w)
        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        c2w = c2w.reshape(b * v, 4, 4)          # 进来就把 b 和 v 合并

        # y - i - 行 - h, x - j - 列 - w
        # y 和 x 都是形状为 [h, w] 的矩阵, y 的每一行都是 0 到 w-1, x 的每一列都是 0 到 h-1
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

        # print(f"y device: {y.device}, x device: {x.device}, image device: {image.device}")
        # print(f"y dtype: {y.dtype}, x dtype: {x.dtype}, image dtype: {image.dtype}")
        # print(f"y shape: {y.shape}, x shape: {x.shape}, image shape: {image.shape}")
        # stx()
        '''
            1. not eval:
                y device: cpu, x device: cpu, image device: cuda:0
                y dtype: torch.int64, x dtype: torch.int64, image dtype: torch.float32
                y shape: torch.Size([256, 256]), x shape: torch.Size([256, 256]), image shape: torch.Size([64, 3, 65536])
            2. eval:
                y device: cpu, x device: cpu, image device: cuda:0
                y dtype: torch.int64, x dtype: torch.int64, image dtype: torch.float32
                y shape: torch.Size([256, 256]), x shape: torch.Size([256, 256]), image shape: torch.Size([64, 3, 65536])
        '''
        y, x = y.to(image.device), x.to(image.device)

        # [h, w], normalize 到 [-1, 1]
        y_norm, x_norm = (y + 0.5) / h * 2 - 1, (x + 0.5) / w * 2 - 1
        # [b, v, 2, h, w]
        xy_norm = torch.stack([x_norm, y_norm], dim=0)[None, None, :, :, :].expand(
            b, v, -1, -1, -1
        )

        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)

        # 到此处, x, y 是图像坐标 (u, v), 它与相机坐标 (x_c, y_c, z_c) 的转换关系是:
        # u = f_x * x_c / z_c + cx
        # v = f_y * y_c / z_c + cy
        # 所以反算 x_c, y_c 是
        # x_c / z_c = (u - cx) / f_x, y_c / z_c = (v - cy) / f_y
        # 那么 per-pixel 的射线方向就是 (x_c, y_c, z_c) = (x_c / z_c, y_c / z_c, 1)

        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)

        # ray_d 是相机坐标系下的射线方向
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d_cam = ray_d.clone()
        # bmm = batch matrix multiplication, 这里转成世界坐标系
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]

        # norm
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_d_cam = ray_d_cam / torch.norm(ray_d_cam, dim=2, keepdim=True)
        
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        # fetch color, ray_o, ray_d for all patch centers
        ray_color_patch, ray_o_patch, ray_d_patch = None, None, None
        ray_xy_norm_patch, proj_mat = None, None
        if patch_size is not None:
            start_patch_center = patch_size / 2.0
            y, x = torch.meshgrid(
                torch.arange(h // patch_size) * patch_size + start_patch_center,
                torch.arange(w // patch_size) * patch_size + start_patch_center,
                indexing="ij",
            )
            y, x = y.to(image.device), x.to(image.device)
            x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
            y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)

            ray_xy_norm_patch = torch.stack(
                [x / w, y / h], dim=2
            )  # use [0,1] for patch center
            K_norm = (
                torch.eye(3, device=image.device).unsqueeze(0).repeat(b * v, 1, 1)
            )  # [b*v, 3, 3]
            K_norm[:, 0, 0] = fxfycxcy[:, 0] / w
            K_norm[:, 1, 1] = fxfycxcy[:, 1] / h
            K_norm[:, 0, 2] = fxfycxcy[:, 2] / w
            K_norm[:, 1, 2] = fxfycxcy[:, 3] / h
            w2c = torch.inverse(c2w)  # [b*v, 4, 4]
            proj_mat = torch.bmm(K_norm, w2c[:, :3, :4])  # [b*v, 3, 4]
            proj_mat = proj_mat.reshape(b * v, 12)
            proj_mat = proj_mat / (proj_mat.norm(dim=1, keepdim=True) + 1e-6)
            proj_mat = proj_mat.reshape(b * v, 3, 4)
            proj_mat = proj_mat * proj_mat[:, 0:1, 0:1].sign()

            # fetch colors for all patch centers
            ray_color_patch = (
                F.grid_sample(
                    image.reshape(b * v, c, h, w),
                    torch.stack([x / w * 2.0 - 1.0, y / h * 2.0 - 1.0], dim=2).reshape(
                        b * v, -1, 1, 2
                    ),
                    align_corners=False,
                )
                .squeeze(-1)
                .permute(0, 2, 1)
            ).contiguous()  # [b*v, h'*w', c]

            x = (x - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
            y = (y - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
            z = torch.ones_like(x)
            ray_d_patch = torch.stack([x, y, z], dim=2)  # [b*v, h'*w', 3]
            ray_d_patch = torch.bmm(
                ray_d_patch, c2w[:, :3, :3].transpose(1, 2)
            )  # [b*v, h'*w', 3]
            ray_d_patch = ray_d_patch / torch.norm(
                ray_d_patch, dim=2, keepdim=True
            )  # [b*v, h'*w', 3]
            ray_o_patch = c2w[:, :3, 3][:, None, :].expand_as(
                ray_d_patch
            )  # [b*v, h'*w', 3]

            n_patch = ray_color_patch.size(1)
            ray_color_patch = ray_color_patch.reshape(b, v, n_patch, c)
            ray_o_patch = ray_o_patch.reshape(b, v, n_patch, 3)
            ray_d_patch = ray_d_patch.reshape(b, v, n_patch, 3)
            ray_xy_norm_patch = ray_xy_norm_patch.reshape(b, v, n_patch, 2)
            proj_mat = proj_mat.reshape(b, v, 3, 4)

        ray_o = ray_o.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
        ray_d = ray_d.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
        ray_d_cam = ray_d_cam.reshape(b, v, h, w, 3).permute(0, 1, 4, 2, 3)
        image = image.reshape(b, v, c, h, w)
        fxfycxcy = fxfycxcy.reshape(b, v, 4)
        c2w = c2w.reshape(b, v, 4, 4)
        
        prepared_input = edict(
                image=image,
                ray_o=ray_o,
                ray_d=ray_d,
                ray_d_cam=ray_d_cam,
                fxfycxcy=fxfycxcy,
                c2w=c2w,
            )
        return prepared_input



'''
    为每个 batch 的输入数据生成网格
    x 坐标和 y 坐标分别存在 x1_list 和 x2_list 中
'''
def batched_meshgrid(x1, x2, indexing="ij"):
    """_summary_

    Args:
        x1 (torch.tensor): [b, n1]
        x2 (torch.tensor): [b, n2]

    Returns:
        x1_grid (torch.tensor): [b, n1, n2]
        x2_grid (torch.tensor): [b, n1, n2]
    """
    x1_list, x2_list = [], []
    for i in range(x1.size(0)):
        x1_i, x2_i = torch.meshgrid(x1[i], x2[i], indexing=indexing)
        x1_list.append(x1_i)
        x2_list.append(x2_i)
    return torch.stack(x1_list, dim=0), torch.stack(x2_list, dim=0)


class TransformTarget(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self, data_batch):
        """Transform target before computing loss

        Args:
            image (torch.tensor): [b, v, c, h, w]
            fxfycxcy (torch.tensor): [b, v, 4]
            c2w (torch.tensor): [b, v, 4, 4]

        Returns:
            target: {"image": (b, v*(crop_size*crop_size+n_rand_rays), 3),
                     "ray_o": (b, v*(crop_size*crop_size+n_rand_rays), 3),
                     "ray_d": (b, v*(crop_size*crop_size+n_rand_rays), 3),
                     "crop_size": crop_size,
                     "n_rand_rays": n_rand_rays}
        """
        image, depth, normal, fxfycxcy, c2w = (
            data_batch["image"],
            data_batch["depth"],            
            data_batch["normal"],
            data_batch["fxfycxcy"],
            data_batch["c2w"],
        )


        crop_size, n_rand_rays, downsample = (
            self.config.training.crop_size,
            self.config.training.n_rand_rays,
            self.config.training.downsample,
        )

        # downsample 表示降采样的 scale factor
        if downsample > 1:
            with torch.cuda.amp.autocast(enabled=False):    # autocast 决定是否启用混合精度
                b, v, c, h, w = image.size()
                dtype = image.dtype
                image = pyrdown(
                    image.reshape(-1, c, h, w).to(dtype=torch.float32),
                    factor=downsample,
                ).to(dtype=dtype)
                _, c, h, w = image.size()
                image = image.reshape(b, v, c, h, w)
                depth = F.interpolate(
                    depth.reshape(-1, 1, h, w).to(dtype=torch.float32),
                    size=(h, w),
                    interpolation="nearest",
                )
                depth = depth.reshape(b, v, 1, h, w)
                normal = F.interpolate(
                    normal.reshape(-1, 3, h, w).to(dtype=torch.float32),
                    size=(h, w),
                    interpolation="nearest",
                )
                normal = normal.reshape(b, v, 3, h, w)

            fxfycxcy = fxfycxcy / downsample

        b, v, c, h, w = image.size()
        image = image.reshape(b * v, c, h * w)
        depth = depth.reshape(b * v, 1, h * w)
        normal = normal.reshape(b * v, 3, h * w)
        
        # sample crop supervision
        # [b*v, crop_size]
        is_crop_performed = (w > crop_size) or (h > crop_size)
        if w > crop_size:
            crop_idx_x = (
                torch.randint(low=0, high=w - crop_size, size=(b * v, 1))
                + torch.arange(crop_size)[None, :]
            ).long()
        else:
            crop_idx_x = torch.arange(w)[None, :].expand(b * v, -1).long()

        if h > crop_size:
            crop_idx_y = (
                torch.randint(low=0, high=h - crop_size, size=(b * v, 1))
                + torch.arange(crop_size)[None, :]
            ).long()
        else:
            crop_idx_y = torch.arange(h)[None, :].expand(b * v, -1).long()

        # modify fxfycxcy
        # 并不影响 focal，只是影响 cx 和 cy
        crop_fxfycxcy = fxfycxcy.clone()  # [b, v, 4]
        if is_crop_performed:
            device = fxfycxcy.device
            crop_fxfycxcy[:, :, 2] = crop_fxfycxcy[:, :, 2] - crop_idx_x[:, 0].reshape(
                b, v
            ).to(
                device
            )  # cx
            crop_fxfycxcy[:, :, 3] = crop_fxfycxcy[:, :, 3] - crop_idx_y[:, 0].reshape(
                b, v
            ).to(
                device
            )  # cy

        # [b*v, crop_size, crop_size]
        # not supported yet
        # crop_idx_y, crop_idx_x = torch.meshgrid(crop_idx_y, crop_idx_x, indexing="ij")
        # batched meshgrid
        crop_idx_y, crop_idx_x = batched_meshgrid(crop_idx_y, crop_idx_x, indexing="ij")
        # [b*v, crop_size*crop_size]
        crop_idx_x, crop_idx_y = crop_idx_x.reshape(b * v, -1), crop_idx_y.reshape(
            b * v, -1
        )

        sample_idx_x, sample_idx_y = crop_idx_x, crop_idx_y
        if n_rand_rays > 0:
            # sample random supervision
            # [b*v, n_rand_rays]
            rand_idx = torch.randint(
                low=0, high=h * w, size=(b * v, n_rand_rays)
            ).long()
            rand_idx_y, rand_idx_x = rand_idx // w, rand_idx % w

            # combine crop and random supervision
            # [b*v, crop_size*crop_size+n_rand_rays]
            sample_idx_x, sample_idx_y = torch.cat(
                [sample_idx_x, rand_idx_x], dim=1
            ), torch.cat([sample_idx_y, rand_idx_y], dim=1)

        sample_idx_x, sample_idx_y = sample_idx_x.to(image.device), sample_idx_y.to(
            image.device
        )
        sample_idx = sample_idx_y * w + sample_idx_x
        # shortcut if no crop and random supervision
        if (not is_crop_performed) and (n_rand_rays == 0):
            ray_color = image  # [b*v, c, h*w]
            ray_depth = depth  # [b*v, 1, h*w]
            ray_normal = normal # [b*v, 3, h*w]
        else:
            ray_color = torch.gather(
                image, dim=2, index=sample_idx[:, None, :].expand(-1, c, -1)
            )  # [b*v, c, crop_size*crop_size+n_rand_rays]
            ray_depth = torch.gather(
                depth, dim=2, index=sample_idx[:, None, :].expand(-1, 1, -1)
            )  # [b*v, 1, crop_size*crop_size+n_rand_rays]
            ray_normal = torch.gather(
                normal, dim=2, index=sample_idx[:, None, :].expand(-1, 3, -1)
            )  # [b*v, 3, crop_size*crop_size+n_rand_rays]
            
        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        c2w = c2w.reshape(b * v, 4, 4)
        x = (sample_idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (sample_idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack(
            [x, y, z], dim=2
        )  # [b*v, crop_size*crop_size+n_rand_rays, 3]

        # 一个 batch 的矩阵乘法，bmm = batch matrix multiplication, 全部转成世界坐标系
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)

        ray_color = (
            ray_color.reshape(b, v, c, -1).permute(0, 1, 3, 2).reshape(b, -1, c)
        )  # [b, v*(crop_size*crop_size+n_rand_rays), c]
        ray_depth = (
            ray_depth.reshape(b, v, 1, -1).permute(0, 1, 3, 2).reshape(b, -1, 1)
        )  # [b, v*(crop_size*crop_size+n_rand_rays), 1]
        ray_normal = (
            ray_normal.reshape(b, v, 3, -1).permute(0, 1, 3, 2).reshape(b, -1, 3)
        )  # [b, v*(crop_size*crop_size+n_rand_rays), 3]
        
        ray_o = ray_o.reshape(b, -1, 3)  # [b, v*(crop_size*crop_size+n_rand_rays), 3]
        ray_d = ray_d.reshape(b, -1, 3)  # [b, v*(crop_size*crop_size+n_rand_rays), 3]
        crop_fxfycxcy = crop_fxfycxcy.reshape(b, -1, 4)  # [b, v, 4]
        c2w = c2w.reshape(b, -1, 4, 4)  # [b, v, 4, 4]
        target = edict(
            ray_color=ray_color.contiguous(),
            ray_o=ray_o.contiguous(),
            ray_d=ray_d.contiguous(),
            batch=b,
            view=v,
            channel=c,
            crop_size=crop_size,
            n_rand_rays=n_rand_rays,
            crop_fxfycxcy=crop_fxfycxcy,
            c2w=c2w,
            ray_depth=ray_depth.contiguous(),
            ray_normal=ray_normal.contiguous(),
        )

        # # debug
        # view_ray_o = ray_o.reshape(b, v, -1, 3)[0, :, :, :]
        # view_ray_d = ray_d.reshape(b, v, -1, 3)[0, :, :, :]
        # from .utils_nerf import sample_pts_uniform

        # is_intersect, xyz, t_vals, dists = sample_pts_uniform(
        #     view_ray_o, view_ray_d, num_samples_per_ray=16, randomize_samples=False
        # )
        # import trimesh
        # import os

        # print(f"debug b,v: {b},{v}")
        # print(f"{view_ray_o[:, 0, :]}")
        # os.makedirs("debug", exist_ok=True)
        # trimesh.PointCloud(view_ray_o.reshape(-1, 3).cpu().numpy()).export(
        #     f"debug/ray_o.ply"
        # )
        # trimesh.PointCloud(
        #     (view_ray_o + 0.2 * view_ray_d).reshape(-1, 3).cpu().numpy()
        # ).export(f"debug/ray_d.ply")
        # for i in range(xyz.shape[0]):
        #     trimesh.PointCloud(xyz[i].reshape(-1, 3).cpu().numpy()).export(
        #         f"debug/xyz_{i}.ply"
        #     )
        #     trimesh.PointCloud(
        #         (view_ray_o + 0.2 * view_ray_d)[i].reshape(-1, 3).cpu().numpy()
        #     ).export(f"debug/ray_d_{i}.ply")

        # exit(0)
        return target


def generate_drop_mask(b, v):
    drop_mask_list = []
    for _ in range(b):
        num_to_drop = random.choice(list(range(v + 1)))
        combinations = list(itertools.combinations(list(range(v)), num_to_drop))
        items_to_drop = random.choice(combinations)

        drop_mask = torch.ones(v, dtype=torch.bool)
        for item in items_to_drop:
            drop_mask[item] = False
        drop_mask_list.append(drop_mask)
    drop_mask_list = torch.stack(drop_mask_list, dim=0)
    return drop_mask_list



class TransformMeshTarget(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def forward(self, data_batch, generate_ray=False):
        """transform input image before feeding into transformer

        Args:
            image (torch.tensor): [b, v, c, h, w]
            fxfycxcy (torch.tensor): [b, v, 4]
            c2w (torch.tensor): [b, v, 4, 4]
            mv (torch.tensor): [b, v, 4, 4]
            mvp (torch.tensor): [b, v, 4, 4]

        Returns:
            image: [b, v, c, h, w]
        """
        image, fxfycxcy, c2w, mv, mvp, depth, normal, index = (
            data_batch.image,
            data_batch.fxfycxcy,
            data_batch.c2w,
            data_batch.mv,
            data_batch.mvp,
            data_batch.depth,
            data_batch.normal,
            data_batch.index,
        )

        assert image.dim() == 5, f"image dim should be 5, but got {image.dim()}"
        assert (
            fxfycxcy.dim() == 3
        ), f"fxfycxcy dim should be 3, but got {fxfycxcy.dim()}"
        assert c2w.dim() == 4, f"c2w dim should be 4, but got {c2w.dim()}"
        assert mv.dim() == 4, f"mv dim should be 4, but got {mv.dim()}"
        assert mvp.dim() == 4, f"mvp dim should be 4, but got {mvp.dim()}"
        assert depth.dim() == 5, f"depth dim should be 5, but got {depth.dim()}"
        assert normal.dim() == 5, f"normal dim should be 5, but got {normal.dim()}"

        crop_size, n_rand_rays, downsample = image.size(3), 0, 1

        b, v, c, h, w = image.size()
        image = image.reshape(b * v, c, h * w)

        # sample crop supervision
        # [b*v, crop_size]
        if w > crop_size:
            crop_idx_x = (
                torch.randint(low=0, high=w - crop_size, size=(b * v, 1))
                + torch.arange(crop_size)[None, :]
            ).long()
        else:
            crop_idx_x = torch.arange(w)[None, :].expand(b * v, -1).long()

        if h > crop_size:
            crop_idx_y = (
                torch.randint(low=0, high=h - crop_size, size=(b * v, 1))
                + torch.arange(crop_size)[None, :]
            ).long()
        else:
            crop_idx_y = torch.arange(h)[None, :].expand(b * v, -1).long()

        # [b*v, crop_size, crop_size]
        # not supported yet
        # crop_idx_y, crop_idx_x = torch.meshgrid(crop_idx_y, crop_idx_x, indexing="ij")
        # batched meshgrid
        crop_idx_y, crop_idx_x = batched_meshgrid(crop_idx_y, crop_idx_x, indexing="ij")
        # [b*v, crop_size*crop_size]
        crop_idx_x, crop_idx_y = crop_idx_x.reshape(b * v, -1), crop_idx_y.reshape(
            b * v, -1
        )
        # sample random supervision
        # [b*v, n_rand_rays]
        rand_idx = torch.randint(low=0, high=h * w, size=(b * v, n_rand_rays)).long()
        rand_idx_y, rand_idx_x = rand_idx // w, rand_idx % w

        # combine crop and random supervision
        # [b*v, crop_size*crop_size+n_rand_rays]
        sample_idx_x, sample_idx_y = torch.cat(
            [crop_idx_x, rand_idx_x], dim=1
        ), torch.cat([crop_idx_y, rand_idx_y], dim=1)
        sample_idx_x, sample_idx_y = sample_idx_x.to(image.device), sample_idx_y.to(
            image.device
        )
        sample_idx = sample_idx_y * w + sample_idx_x
        ray_color = torch.gather(
            image, dim=2, index=sample_idx[:, None, :].expand(-1, c, -1)
        )  # [b*v, c, crop_size*crop_size+n_rand_rays]

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        c2w = c2w.reshape(b * v, 4, 4)
        x = (sample_idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (sample_idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack(
            [x, y, z], dim=2
        )  # [b*v, crop_size*crop_size+n_rand_rays, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)

        ray_color = (
            ray_color.reshape(b, v, c, -1).permute(0, 1, 3, 2).reshape(b, -1, c)
        )  # [b, v*(crop_size*crop_size+n_rand_rays), c]
        ray_o = ray_o.reshape(b, -1, 3)  # [b, v*(crop_size*crop_size+n_rand_rays), 3]
        ray_d = ray_d.reshape(b, -1, 3)  # [b, v*(crop_size*crop_size+n_rand_rays), 3]

        image = image.reshape(b, v, c, h, w)
        fxfycxcy = fxfycxcy.reshape(b, v, 4)
        c2w = c2w.reshape(b, v, 4, 4)
        input = edict(
            image=image,
            fxfycxcy=fxfycxcy,
            c2w=c2w,
            mv=mv,
            mvp=mvp,
            depth=depth,
            normal=normal,
            index=index,
            ray_o=ray_o,
            ray_d=ray_d,
        )
        return input