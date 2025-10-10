import numpy as np
import torch
import torch.nn as nn
import json
from diffusers import DDIMScheduler

from diffusionGS.utils.typing import *
from tqdm import tqdm
import math
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusionGS.utils.structure import Mesh
import inspect
from PIL import Image, ImageOps
import rembg
import cv2
def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "scaled_linear":
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64)**2
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = next(
        (
            obj
            for obj in (mean1, logvar1, mean2, logvar2)
            if isinstance(obj, torch.Tensor)
        ),
        None,
    )
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x, device=tensor.device)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def euler_sample(ddim_scheduler,
                diffusion_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True):

    assert steps > 0, f"{steps} must > 0."

    # init latents
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        bsz = bsz // 2

    # latents = torch.randn(
    #     (bsz, *shape),
    #     generator=generator,
    #     device=cond.device,
    #     dtype=cond.dtype,
    # )
    latents = diffusion_model.get_init_point(bsz)
    #latents = latents + init_offsets
    # scale the initial noise by the standard deviation required by the scheduler
    #latents = latents * ddim_scheduler.init_noise_sigma
    # set timesteps
    ddim_scheduler.set_timesteps(steps)
    timesteps = ddim_scheduler.timesteps.to(device)
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, and between [0, 1]
    extra_step_kwargs = {
        # "eta": eta,
        "generator": generator
    }
    
    #timesteps, num_inference_steps = retrieve_timesteps(ddim_scheduler, steps, device)
    #breakpoint()
    # reverse
    for i, t in enumerate(tqdm(timesteps, disable=disable_prog, desc="Euler Sampling:", leave=False)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if do_classifier_free_guidance
            else latents
        )
        #breakpoint()
        # predict the noise residual
        timestep_tensor = torch.tensor(t.clone().detach(), dtype=torch.long, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model.forward(latent_model_input, timestep_tensor, cond, train_stage = False)

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        latents = ddim_scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs
        ).prev_sample

        yield latents, t


def compute_snr(alphas_cumprod, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def get_val_data(file_list, n_supervision=4096, geometry_type="occupancies"):
    batch_rand_points = []
    batch_occupancies = []
    for file in file_list:
        points = np.load(file)
        rand_points = np.asarray(points['points']) * 2 # range from -1.1 to 1.1
        rand_points = torch.from_numpy(rand_points)
        rand_points = torch.split(rand_points, n_supervision, dim=0)
        rand_points = torch.stack(rand_points[0:-1])
        occupancies = np.asarray(points[geometry_type])
        occupancies = np.unpackbits(occupancies)
        occupancies = torch.from_numpy(occupancies)
        occupancies = torch.split(occupancies, n_supervision, dim=0)
        occupancies = torch.stack(occupancies[0:-1])
        batch_rand_points.append(rand_points)
        batch_occupancies.append(occupancies)
    batch_rand_points = torch.stack(batch_rand_points)
    batch_occupancies= torch.stack(batch_occupancies)
    B, M, N, _ = batch_rand_points.shape
    return batch_rand_points.view(B*M, N, 3), batch_occupancies.view(B*M, N)

def get_mvp_matrix(c2w:Tensor, proj_mtx:Tensor):
    '''
    c2w: [batch_size, 4, 4]
    proj_mtx: [batch_size, 4, 4]

    mvp_mtx: [batch_size, 4, 4]
    '''
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx.to(c2w)

def get_projection_matrix_perspective(intrinsics:Tensor, near: float = 0.01, far: float = 100.0):
    '''
    intrinsics: [batch_size, 3, 3], normalized

    proj_mtx: [batch_size, 4, 4]
    '''
    batch_size = intrinsics.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32).to(intrinsics)
    proj_mtx[:, 0, 0] = 2 * intrinsics[:, 0, 0]
    proj_mtx[:, 1, 1] = - 2 * intrinsics[:, 1, 1] # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def render_mesh(mesh, cam2world_matrices, intrinsics, device, height=224, width=224, radius=1):
    import nvdiffrast.torch as dr

    ## world to ndc
    v_homo = torch.cat([mesh.v_pos, torch.ones([mesh.v_pos.shape[0], 1]).to(mesh.v_pos)], dim=-1)
    mvp_mtx = get_mvp_matrix(cam2world_matrices, get_projection_matrix_perspective(intrinsics))
    v_pos_clip = torch.matmul(v_homo, mvp_mtx.permute(0, 2, 1)).float()

    v_nrm = mesh.v_nrm
    
    ## rasterize
    ctx = dr.RasterizeCudaContext(device)
    mesh.t_pos_idx = mesh.t_pos_idx.to(torch.int32)
    rast, _ = dr.rasterize(ctx, v_pos_clip, mesh.t_pos_idx, (height, width))
    mask = rast[..., 3:] > 0

    ## render alpha
    mask_aa = dr.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

    ## render depth
    gb_depth, _ = dr.interpolate(v_pos_clip[:, :, 3:].contiguous(), rast, mesh.t_pos_idx)
    gb_depth_aa = torch.lerp(torch.zeros_like(gb_depth), gb_depth, mask.float())
    # gb_depth_aa = dr.antialias(gb_depth_aa, rast, v_pos_clip, mesh.t_pos_idx)

    ## render normal
    gb_normal, _ = dr.interpolate(v_nrm, rast, mesh.t_pos_idx)
    gb_normal = F.normalize(gb_normal, dim=-1)
    gb_normal_aa = torch.lerp(torch.full_like(gb_normal, fill_value=-1.0), gb_normal, mask.float())
    gb_normal_aa = dr.antialias(gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx)

    ## render rgb
    selector = mask[..., 0]
    gb_pos, _ = dr.interpolate(mesh.v_pos, rast, mesh.t_pos_idx)
    positions = gb_pos[selector]
    gb_rgb_fg = torch.zeros(*mask_aa.shape[:3], 3).to(device)
    gb_rgb_fg[selector] = torch.ones(3).to(device) * 125
    gb_rgb_bg = torch.ones_like(gb_rgb_fg).to(gb_rgb_fg)
    gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.to(gb_rgb_fg))
    gb_rgb_aa = dr.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)
    # out["rgb"] = gb_rgb_aa

    out = {
        "alpha": mask_aa,
        "depth": gb_depth_aa,
        "rgb": gb_rgb_aa,
        "normal": gb_normal_aa
    }
    return out

def get_camera(device, img_size=224, focal=5, distance=10):
    cam2world_matrices = torch.as_tensor([
                                [[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, distance],
                                [0, 0, 0, 1]], # front to back

                                [[0, 0, 1, distance],
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]], # front to back

                                [[-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, -1, -distance],
                                [0, 0, 0, 1]], # front to back

                                [[0, 0, -1, -distance],
                                [0, 1, 0, 0],
                                [-1, 0, 0, 0],
                                [0, 0, 0, 1]], # front to back
                            ], dtype=torch.float32).to(device)
    intrinsics = torch.zeros([3, 3]).to(device)
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[1, 2] = img_size // 2
    intrinsics[0, 2] = img_size // 2
    intrinsics = intrinsics.unsqueeze(0).repeat(4, 1 ,1).to(torch.float32)

    return cam2world_matrices, intrinsics

def read_image(img, img_size=224):
    transform = transforms.Compose(
            [
                transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(img_size),  # crop a (224, 224) square
                transforms.ToTensor()
            ]
        )
    rgb = Image.open(img)
    rgb = transform(rgb)[:3,...].permute(1, 2, 0)
    return rgb

def compute_metric(shape_model, sample_inputs, sample_outputs, device):
    threshold = 0
    if len(sample_inputs['occupancy']) == 0:
        return 0, 0
    queries, labels = get_val_data(sample_inputs['occupancy'])
    queries = queries.to(device)
    labels = labels.to(device)
    latent = sample_outputs[0][:len(sample_inputs['occupancy']),...]
    latent = latent.unsqueeze(1).repeat(1, queries.shape[0] // latent.shape[0], 1, 1).view(queries.shape[0], latent.shape[1], latent.shape[2])
    outputs = shape_model.query(queries, latent)
    pred = torch.zeros_like(outputs)
    pred[outputs>=threshold] = 1
    torch.cuda.empty_cache()
    accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
    accuracy = accuracy.mean()
    intersection = (pred * labels).sum(dim=1)
    union = (pred + labels).gt(0).sum(dim=1)
    iou = intersection * 1.0 / union + 1e-5
    iou = iou.mean()

    return accuracy, iou

def compute_metric_from_sdf(shape_model, sample_inputs, sample_outputs, threshold, device):
    if len(sample_inputs['sdf']) == 0:
        return 0, 0
    queries, sdf = get_val_data(sample_inputs['sdf'])
    queries = queries.to(device)
    labels = labels.to(device)
    latent = sample_outputs[0][:len(sample_inputs['occupancy']),...]
    latent = latent.unsqueeze(1).repeat(1, queries.shape[0] // latent.shape[0], 1, 1).view(queries.shape[0], latent.shape[1], latent.shape[2])
    outputs = shape_model.query(queries, latent)
    pred = torch.zeros_like(outputs)
    pred[outputs>=threshold] = 1
    torch.cuda.empty_cache()
    accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
    accuracy = accuracy.mean()
    intersection = (pred * labels).sum(dim=1)
    union = (pred + labels).gt(0).sum(dim=1)
    iou = intersection * 1.0 / union + 1e-5
    iou = iou.mean()

    return accuracy, iou

def get_render(vertices, faces, device, img_size=224, focal=5, distance=12):
    cam2world_matrices, intrinsics = get_camera(device, img_size, focal, distance)
    mesh = Mesh(
                v_pos=torch.from_numpy(vertices).to(device),
                t_pos_idx=torch.from_numpy(faces).to(device),
            )
    out = render_mesh(mesh, cam2world_matrices, intrinsics, device, img_size, img_size)
    v, h, w, _ = out['normal'].shape
    normal = out['normal'].transpose(1, 0).contiguous().view(h, w*v, 3)
    depth = out['depth'].transpose(1, 0).contiguous().view(h, w*v, 1)
    if depth[depth != 0].shape[0] != 0:
        depth[depth != 0] = (depth[depth != 0] - depth[depth != 0].min()) / (depth[depth != 0].max() - depth[depth != 0].min())

    return normal, depth


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def get_bbox(mask:np.ndarray):
    assert mask.ndim == 2
    row = mask.sum(-1)
    col = mask.sum(-2)

    row_idx = np.where(row > 0)[0]
    col_idx = np.where(col > 0)[0]
    x1, y1, x2, y2 = col_idx.min(), row_idx.min(), col_idx.max(), row_idx.max()
    return np.array([x1, y1, x2, y2])


def preprocess(image:Image.Image,session) -> Image.Image:
    H_1 = W_1 = 512
    scale = 0.8
    color = 'white'
    image = ImageOps.exif_transpose(image)
    if image.mode == 'RGBA' and np.sum(np.array(image.getchannel('A')) > 0) < image.size[0] * image.size[1] - 8:
        rgba = image.copy()
    else:
        # https://github.com/pymatting/pymatting/issues/19
        rgba = rembg.remove(image, alpha_matting=True, session=session)

    alpha = rgba.getchannel('A')
    bboxs = get_bbox(np.array(alpha))
    x1, y1, x2, y2 = bboxs
    dy, dx = y2 - y1, x2 - x1
    s = min(H_1 * scale / dy, W_1 * scale / dx)
    Ht, Wt = int(dy * s), int(dx * s)
    ox, oy = int((W_1 - Wt) / 2), int((H_1 - Ht) / 2)
    bboxt = np.array([ox, oy, ox+Wt, oy+Ht])

    rgba = rgba.crop(bboxs).resize((Wt, Ht))
    fg = rgba.convert('RGB')
    alpha = rgba.getchannel('A')
    alphat = Image.new('L', (W_1, H_1))
    alphat.paste(alpha, bboxt)

    inp_1 = Image.new('RGBA', (W_1, H_1), color)
    inp_1.paste(fg, bboxt, alpha)
    inp_1.putalpha(alphat)
    return inp_1


# def preprocess(image:Image.Image,session) -> Image.Image:
#     H_1 = W_1 = 512
#     color = 'white'
#     image = ImageOps.exif_transpose(image)
#     if image.mode == 'RGBA' and np.sum(np.array(image.getchannel('A')) > 0) < image.size[0] * image.size[1] - 8:
#         rgba = image.copy()
#     else:
#         # https://github.com/pymatting/pymatting/issues/19
#         rgba = rembg.remove(image, alpha_matting=True, session=session)


#     fg = rgba.convert('RGB')
#     alpha = rgba.getchannel('A')
#     alphat = Image.new('L', (W_1, H_1))
#     alphat.paste(alpha)

#     inp_1 = Image.new('RGBA', (W_1, H_1), color)
#     inp_1.paste(fg, mask = alpha)
#     inp_1.putalpha(alphat)
#     return inp_1


def TransformInput(image, c2w, fxfycxcy, patch_size=None):
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
    ray_d = torch.bmm(ray_d.to(c2w), c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]

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
        proj_mat = torch.bmm(K_norm.to(w2c), w2c[:, :3, :4])  # [b*v, 3, 4]
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
            ray_d_patch.to(c2w), c2w[:, :3, :3].transpose(1, 2)
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
    return ray_o, ray_d


# 根据 timestep 把 x_t 做成 video 的可视化代码
def display_timestep_on_video(frames, timesteps):
    """
    frames: [T, H, W, C]
    timesteps: [T, ]
    """
    assert len(frames) == len(
        timesteps
    ), f"len(frames) {len(frames)} != len(timesteps) {len(timesteps)}"

    dtype = frames.dtype
    if dtype == np.float32:
        frames = (frames * 255.0).clip(0.0, 255.0).astype(np.uint8)

    # Use putText() method for
    # inserting text on video
    for i in range(len(frames)):
        # stx()
        frames[i] = cv2.putText(
            frames[i],
            f"t={timesteps[i]}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_4,
        )

    if dtype == np.float32:
        frames = frames.astype(np.float32) / 255.0

    return frames
