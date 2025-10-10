# Adapted from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478
import os

import scipy.io
import torch
import torch.nn as nn

import torch.nn as nn
from pytorch_msssim import SSIM
import lpips
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from einops import reduce, rearrange
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.max1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.max2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.max3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu12 = nn.ReLU(inplace=True)
        self.max4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu14 = nn.ReLU(inplace=True)

        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu15 = nn.ReLU(inplace=True)

        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu16 = nn.ReLU(inplace=True)
        self.max5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, return_style):
        out1 = self.conv1(x)
        out2 = self.relu1(out1)

        out3 = self.conv2(out2)
        out4 = self.relu2(out3)
        out5 = self.max1(out4)

        out6 = self.conv3(out5)
        out7 = self.relu3(out6)
        out8 = self.conv4(out7)
        out9 = self.relu4(out8)
        out10 = self.max2(out9)
        out11 = self.conv5(out10)
        out12 = self.relu5(out11)
        out13 = self.conv6(out12)
        out14 = self.relu6(out13)
        out15 = self.conv7(out14)
        out16 = self.relu7(out15)
        out17 = self.conv8(out16)
        out18 = self.relu8(out17)
        out19 = self.max3(out18)
        out20 = self.conv9(out19)
        out21 = self.relu9(out20)
        out22 = self.conv10(out21)
        out23 = self.relu10(out22)
        out24 = self.conv11(out23)
        out25 = self.relu11(out24)
        out26 = self.conv12(out25)
        out27 = self.relu12(out26)
        out28 = self.max4(out27)
        out29 = self.conv13(out28)
        out30 = self.relu13(out29)
        out31 = self.conv14(out30)
        out32 = self.relu14(out31)

        if return_style > 0:
            return [out2, out7, out12, out21, out30]
        else:
            return out4, out9, out14, out23, out32


class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu") -> None:
        super().__init__()
        self.Net = VGG19()

        weight_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "imagenet-vgg-verydeep-19.mat")
        assert os.path.isfile(
            weight_file
        ), f"Run: wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat \n md5sum imagenet-vgg-verydeep-19.mat == 106118b7cf60435e6d8e04f6a6dc3657 (https://www.vlfeat.org/matconvnet/pretrained/)"

        #weight_file = "/sensei-fs/users/kaiz/repos/weight-collections/imagenet-vgg-verydeep-19.mat"

        vgg_rawnet = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_rawnet["layers"][0]
        layers = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        att = [
            "conv1",
            "conv2",
            "conv3",
            "conv4",
            "conv5",
            "conv6",
            "conv7",
            "conv8",
            "conv9",
            "conv10",
            "conv11",
            "conv12",
            "conv13",
            "conv14",
            "conv15",
            "conv16",
        ]
        S = [
            64,
            64,
            128,
            128,
            256,
            256,
            256,
            256,
            512,
            512,
            512,
            512,
            512,
            512,
            512,
            512,
        ]
        for L in range(16):
            getattr(self.Net, att[L]).weight = nn.Parameter(
                torch.from_numpy(vgg_layers[layers[L]][0][0][2][0][0]).permute(
                    3, 2, 0, 1
                )
            )
            getattr(self.Net, att[L]).bias = nn.Parameter(
                torch.from_numpy(vgg_layers[layers[L]][0][0][2][0][1]).view(S[L])
            )
        self.Net = self.Net.eval().to(device)
        for param in self.Net.parameters():
            param.requires_grad = False

    def compute_error(self, truth, pred):
        E_per_element = torch.abs(truth - pred)   # [32, 3, 256, 256]
        b, c, h, w = E_per_element.shape
        # stx()
        E = torch.mean(E_per_element, dim=tuple(range(1, E_per_element.dim())))
        return E

    def forward(self, pred_img, real_img):
        """
        pred_img, real_img: [B, 3, H, W] in range [0, 1]
        """
        bb = (
            torch.Tensor([123.6800, 116.7790, 103.9390])
            .float()
            .reshape(1, 3, 1, 1)
            .to(pred_img.device)
        )

        real_img_sb = real_img * 255.0 - bb
        pred_img_sb = pred_img * 255.0 - bb

        out3_r, out8_r, out13_r, out22_r, out33_r = self.Net(
            real_img_sb, return_style=0
        )
        out3_f, out8_f, out13_f, out22_f, out33_f = self.Net(
            pred_img_sb, return_style=0
        )

        E0 = self.compute_error(real_img_sb, pred_img_sb)
        E1 = self.compute_error(out3_r, out3_f) / 2.6
        E2 = self.compute_error(out8_r, out8_f) / 4.8
        E3 = self.compute_error(out13_r, out13_f) / 3.7
        E4 = self.compute_error(out22_r, out22_f) / 5.6
        E5 = self.compute_error(out33_r, out33_f) * 10 / 1.5

        total_loss = (E0 + E1 + E2 + E3 + E4 + E5) / 255.0
        return total_loss
    


class SsimLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range
        self.ssim_module = SSIM(
            win_size=11,
            win_sigma=1.5,
            data_range=self.data_range,
            size_average=False,
            channel=3,
        )

    def forward(self, x, y):
        """
        x: (N, C, H, W)
        y: (N, C, H, W)
        """
        # stx()
        return 1 - self.ssim_module(x, y)





class LossComputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips_loss_module = lpips.LPIPS(net="vgg")
        self.lpips_loss_module.eval()
        # freeze the lpips loss module; is this correct?
        for param in self.lpips_loss_module.parameters():
            param.requires_grad = False

        # self.perceptual_loss_module = PerceptualLoss()
        # self.perceptual_loss_module.eval()
        # # freeze the perceptual loss module
        # for param in self.perceptual_loss_module.parameters():
        #     param.requires_grad = False

        self.ssim_loss_module = SsimLoss()
        self.ssim_loss_module.eval()
        # freeze the ssim loss module
        for param in self.ssim_loss_module.parameters():
            param.requires_grad = False

    def forward(
        self,
        rendering,
        target,
        masks_all,
        masks,
        ray_o,
        img_aligned_xyz=None,
        gt_img_aligned_xyz=None,
    ):
        """
        rendering: [b, v, 3, h, w]; in range (0, 1)
        target: [b, v, 3, h, w]; in range (0, 1)
        """
        b, v, _, h, w = rendering.size()
        rendering = rendering.reshape(b * v, -1, h, w)
        target = target.reshape(b * v, -1, h, w)

        mask = None
        if target.size(1) == 4:
            target, mask = target.split([3, 1], dim=1)      # rgb + mask
        # image scale rendering loss
        l2_loss = torch.ones(b, device=rendering.device) * 1e-8
        per_element_loss = F.mse_loss(rendering, target, reduction='none')
        per_element_loss = per_element_loss.reshape(b, v, -1, h, w)
        l2_loss = per_element_loss.mean(dim=tuple(range(1, per_element_loss.dim())))
        ## image pixel aligined xyz loss
        if img_aligned_xyz is not None and gt_img_aligned_xyz is not None:
            l2_loss_xyz = torch.ones(b, device=rendering.device) * 1e-8
            per_element_loss_xyz = F.mse_loss(img_aligned_xyz * masks, gt_img_aligned_xyz * masks, reduction='sum')
            l2_loss_xyz = per_element_loss_xyz/masks.sum()
        else:
            l2_loss_xyz = torch.zeros_like(l2_loss)
        ## depth loss
        # l2_loss_depth = torch.ones(b, device=rendering_depth.device) * 1e-8
        # per_element_loss_depth = F.mse_loss(rendering_depth * masks_all, target_depth * masks_all, reduction='sum')
        # l2_loss_depth = per_element_loss_depth/masks_all.sum()
        #l2_loss_xyz = per_element_loss_xyz.mean(dim=tuple(range(1, per_element_loss_xyz.dim())))
        # stx()

        # breakpoint()
        psnr = -10.0 * torch.log10(l2_loss)
        lpips_loss = torch.zeros(b, device=l2_loss.device)
        lpips_loss_per_element = self.lpips_loss_module(
            F.interpolate(rendering, size=[256,256], mode='bilinear') * 2.0 - 1.0, F.interpolate(target, size=[256,256], mode='bilinear') * 2.0 - 1.0
        )
        # stx()
        #lpips_loss_per_element = lpips_loss_per_element.reshape(b, v, 1, 1, 1)
        lpips_loss = lpips_loss_per_element.mean()

        # perceptual_loss = torch.zeros(b, device=l2_loss.device)
        # perceptual_loss = self.perceptual_loss_module(rendering, target)
        # # stx()
        # perceptual_loss = perceptual_loss.reshape(b, v)
        # perceptual_loss = perceptual_loss.mean(dim=1)

        ssim_loss = torch.zeros(b, device=l2_loss.device)
        ssim_loss = self.ssim_loss_module(rendering, target)
        # stx()
        ssim_loss = ssim_loss.reshape(b, v)
        ssim_loss = ssim_loss.mean(dim=1)

        pointsdist_loss = torch.zeros(b, device=l2_loss.device)
        # config 里面是 0, 此处不执行
            # compute target mean and std distance
        trgt_mean_dist = torch.norm(
        ray_o, dim=2, p=2, keepdim=True
            )  # [b, v, 1, h, w]
            # trgt_width = 1.0

            # # compute predicted distance
            # dist = (img_aligned_xyz - input.ray_o).norm(
            #     dim=2, p=2, keepdim=True
            # )  # [b, v, 1, h, w]

            # # compute target distance
            # b, v, _, h, w = dist.size()
            # dist_detach = dist.detach().reshape(b, v, -1)
            # dist_mean = dist_detach.mean(dim=2)[..., None, None, None]
            # dist_max = dist_detach.max(dim=2).values[..., None, None, None]
            # dist_min = dist_detach.min(dim=2).values[..., None, None, None]
            # dist_detach = dist_detach.reshape(b, v, 1, h, w)
            # trgt_dist = (dist_detach - dist_mean) / (
            #     dist_max - dist_min + 1e-8
            # ) * trgt_width + trgt_mean_dist

        trgt_std_dist = 0.5

        # compute predicted distance
        # breakpoint()
        dist = (img_aligned_xyz - ray_o).norm(
                dim=2, p=2, keepdim=True
            )  # [b, v, 1, h, w]          r = o + td, 这里计算到 o 的距离

            # compute target distance
        dist_detach = dist.detach()
        dist_mean = dist_detach.mean(dim=(2, 3, 4), keepdim=True)       # 到 o 的平均距离
        dist_std = dist_detach.std(dim=(2, 3, 4), keepdim=True)         # 到 o 的距离的标准差
        trgt_dist = (dist_detach - dist_mean) / (
            dist_std + 1e-8
        ) * trgt_std_dist + trgt_mean_dist          # 希望到 o 的距离是一个标准正态分布

        pointsdist_loss = (dist - trgt_dist) ** 2
        pointsdist_loss = pointsdist_loss.mean(dim=tuple(range(1, pointsdist_loss.dim())))
            # stx()
        # 不要沿着 batch 去算 average loss, 因为 batch 与 timestep 对应
        # 后边在外包的 diffusion 会算均值
        # stx()
        return l2_loss, lpips_loss, ssim_loss, pointsdist_loss, l2_loss_xyz#, l2_loss_depth



class MetricComputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips_loss_module = lpips.LPIPS(net="vgg")
        self.lpips_loss_module.eval()
        # freeze the lpips loss module; is this correct?
        for param in self.lpips_loss_module.parameters():
            param.requires_grad = False
        

    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth,
        predicted,
    ):
        """
        Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
        
        Args:
            ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
            predicted: Images with shape [batch, channel, height, width], values in [0, 1]
            
        Returns:
            PSNR values for each image in the batch
        """
        ground_truth = torch.clamp(ground_truth, 0, 1)
        predicted = torch.clamp(predicted, 0, 1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * torch.log10(mse) 

    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth,
        predicted,
    ):
        """
        Compute Learned Perceptual Image Patch Similarity between images.
        
        Args:
            ground_truth: Images with shape [batch, channel, height, width]
            predicted: Images with shape [batch, channel, height, width]
            The value range is [0, 1] when we have set the normalize flag to True.
            It will be [-1, 1] when the normalize flag is set to False.
        Returns:
            LPIPS values for each image in the batch (lower is better)
        """
        lpips_loss_per_element = self.lpips_loss_module(
            F.interpolate(predicted, size=[256,256], mode='bilinear') * 2.0 - 1.0, F.interpolate(ground_truth, size=[256,256], mode='bilinear') * 2.0 - 1.0
        )
        return lpips_loss_per_element.squeeze()



    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth,
        predicted,
    ):
        """
        Compute Structural Similarity Index between images.
        
        Args:
            ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
            predicted: Images with shape [batch, channel, height, width], values in [0, 1]
            
        Returns:
            SSIM values for each image in the batch (higher is better)
        """
        ssim_values= []
        
        for gt, pred in zip(ground_truth, predicted):
            # Move to CPU and convert to numpy
            gt_np = gt.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            
            # Calculate SSIM
            ssim = structural_similarity(
                gt_np,
                pred_np,
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            ssim_values.append(ssim)
        
        # Convert back to tensor on the same device as input
        return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)



    def forward(self, target, rendering):
        rendering = rendering.reshape(-1,rendering.shape[-3],rendering.shape[-2],rendering.shape[-1])
        target = target.reshape(-1,target.shape[-3],target.shape[-2],target.shape[-1])
        psnr = self.compute_psnr(target, rendering)
        ssim = self.compute_ssim(target, rendering)
        lpips = self.compute_lpips(target, rendering)
        return psnr, ssim, lpips