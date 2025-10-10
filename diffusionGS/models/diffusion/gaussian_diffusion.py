# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch as th
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl
from pdb import set_trace as stx
import copy
from tqdm import tqdm
import os
from einops import rearrange
from PIL import Image


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


'''
    ModelMeanType 是一个枚举类, 里面有三个枚举值, 可以通过属性的方式来访问, 如 ModelMeanType.PREVIOUS_X
    enum.auto() 会自动分配一个值, 从 1 开始, 依次递增
'''
class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}, 从 x_t 预测 x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


'''
    Gaussian 的 variance 不应该是 schedule 的吗?
'''
class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


# num_diffusion_timesteps = T
# warmup_frac - 指的是 warmup 的比例
# beta 逐步变化，最后到一个 beta_end 后保持不变
def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(
        beta_start, beta_end, warmup_time, dtype=np.float64
    )
    return betas



# * 之前位置参数, * 之后必须是关键字参数
# 不同的 beta_schedule 策略
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
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


# 另一些之前被命名过的 beta scheduler
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


# 还是一个 scheduler
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


# 注意，这只是个 general 的框架, model 是灵活的
'''
    如果想要 x_0 output 的话，好像就只用在调用 GaussianDiffusion 时的时候，设置 model_mean_type = ModelMeanType.START_X 即可
    因为 x_{t-1} 一直是拿 x_t 和 x_0 来算的, q(x_{t-1} | x_t, x_0)
'''
class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type):

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Use float64 for accuracy.
        '''
            beta 控制的是前向传播过程中逐步所加的噪声的方差
            q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t)
            alpha_t = 1 - beta_t
        '''
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D, [B,]"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        # cumulative product of alphas over time
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # 在 self.alphas_cumprod 数组的开始处添加了一个元素 1.0，并且移除了最后一个元素
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # 在 self.alphas_cumprod 数组的末尾添加了一个元素 0.0，并且移除了第一个元素
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)         # reciprocal - 倒数
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # 起始值为 0 不可计算, 只可计算后续的值
        self.posterior_log_variance_clipped = (
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
            if len(self.posterior_variance) > 1
            else np.array([])
        )

        '''
            下边两个是用来计算 q(x_{t-1} | x_t, x_0) 均值的两个系数
            coef1 是 x_0 的系数, coef2 是 x_t 的系数
            然后 x_0 是由 x_t 和 predict 的 噪声 eps 线性组合得到的, 是模型这个阶段猜出的 x_0, 并不是真正的 x_0
        '''
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    '''
        q(x_t | x_0) = N(x_t; sqrt(alpha_t_bar) x_0, sqrt(1 - alpha_t_bar))
    '''
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs. 真实的 x_0, 非模型所猜
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start   # sqrt(alpha_t_bar) x_0
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)    # sqrt(1 - alpha_t_bar)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    # x_0, t -> x_t
    # q(x_t | x_0) = N(x_t; sqrt(alpha_t_bar) x_0, sqrt(1 - alpha_t_bar))
    # 注意：此处不是重参数化采样 ！！ 因为此处不需要求导了, x_t 是U-Net 的输入
    # 重参数化采样发生在 q(x_{t-1} | x_t, x_0), 因为此时就需要求导，loss在 eps 上计算，直接关联 x_{t-1} 的 μ 和 σ, 需要梯度回传
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )


    # prior - 啥也没观测到时，事件A的概率 p(A)
    # posterior - 观测到一些数据或者结果 B 后, A事件发生的概率 p(A|B)
    # 此处的 x_start 是模型猜的 x_0
    # 由于此时 U-Net 已经完成了 eps 的预测，所以这里没有 model
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        # stx()
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    # 此处调用 model
    def p_mean_variance(self, model, input_batch, t, clip_denoised=True, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        x = input_batch["image_noisy"]

        B, V, C = x.shape[:3]
        assert t.shape == (B,)
        '''
            input_batch: dict_keys(['image', 'c2w', 'fxfycxcy', 'index', 'depth', 'normal', 'image_noisy'])
            image: [8, 8, 3, 256, 256]
            image_noisy: [8, 3, 3, 256, 256]
            model_output: [8, 3, 3, 256, 256]
            这里存一组图片来看一下
        '''
        # stx()
        if t[0] > 0:
            input_batch['image'] = th.cat([input_batch['image'][:,0:1],input_batch['image_noisy']],dim=1) # select first images and update input
            render_imgs, pred_guassians = model(input_batch, t)
            #render_imgs, pred_guassians = model(**input_batch)
            model_output = render_imgs[:,1:]
            output_dict = {
                'render_images':render_imgs,
                'pred_gaussians':pred_guassians,
            }
        else:
            input_batch['image'] = th.cat([input_batch['image'][:,0:1],input_batch['image_noisy']],dim=1) # select first images and update input
            render_imgs, pred_guassians = model(input_batch, t)
            model_output = render_imgs[:,1:]
            output_dict = {
                'render_images':render_imgs,
                'pred_gaussians':pred_guassians,
            }
        # variance 的两种方式，一是模型直接输出
        # 二是预先设定好的 variance (beta) 直接按照 timestep 取值
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)


        # 为什么要 clamp 到 [-1, 1]?
        def process_xstart(x):
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            # stx()
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        
        # q(x_{t-1} | x_t, x_0), 这里的 x_0 是模型猜的
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )


        ######################################################################################################################
        ###########################################    sampling 存图 debug     ###############################################
        ######################################################################################################################

        # sampling_debug_path = os.path.join("/sensei-fs/users/yuanhaoc/gslrm_dit/minLRM/debug_sampling", "gaussian_diffusion")
        # os.makedirs(sampling_debug_path, exist_ok=True)

        # im = model_output
        # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
        # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
        # Image.fromarray(im).save(os.path.join(sampling_debug_path, "render_input.png"))

        # im = input_batch["image_noisy"]
        # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
        # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
        # Image.fromarray(im).save(os.path.join(sampling_debug_path, "xt_noisy.png"))

        # im = model_mean
        # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
        # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
        # Image.fromarray(im).save(os.path.join(sampling_debug_path, "model_mean_next_t.png"))

        # im = model_variance
        # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
        # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
        # Image.fromarray(im).save(os.path.join(sampling_debug_path, "model_variance_next_t.png"))

        # im = pred_xstart
        # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
        # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
        # Image.fromarray(im).save(os.path.join(sampling_debug_path, "pred_xstart_after_clipping.png"))

        # stx()


        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,                     # μθ of x_{t-1}
            "variance": model_variance,             # σ^2
            "log_variance": model_log_variance,     # log(σ^2)
            "pred_xstart": pred_xstart,             # x_0
            "denoiser_output_dict": output_dict,
        }

    # 从 eps 预测 x_0, 正常走的是这一步
    # x_0 = 1/sqrt(alpha_t_bar) x_t - sqrt((1-alpha_t_bar) / alpha_t_bar) eps
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    # 由 x_0 预测 eps, 这里假设模型输出 x_0
    # eps = (x_t - sqrt(alpha_t_bar) x_0) / sqrt(1-alpha_t_bar)
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # predict x_{t-1}
    def p_sample(
        self,
        model,
        input_batch,
        t,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # stx()
        x = input_batch["image_noisy"]
        out = self.p_mean_variance(
            model,
            input_batch,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # 此处是重参数化采样 x_{t-1} = μ + σ * eps
        # [1] 为何要 nonzero_mask, 因为底 0 步可以直接输出 x_0, 不需要加噪声
        # [2] 为何要有一个 torch.exp(0.5), 给方差开根号就是协方差
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        input_batch["image_noisy"] = sample
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "input_batch": input_batch, "denoiser_output_dict": out["denoiser_output_dict"]} # 返回 x_{t-1}, x_0, 和 input_batch

    def p_sample_loop(
        self,
        model,
        shape,
        input_batch=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=True,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        # 对 self.p_sample_loop_progressive 函数返回的对象进行 forloop
        # stx()
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            input_batch=input_batch,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample          # 此处对应的 sample 应该是 p_sample 函数的 return 值，但这个函数好像就只需要里面的 "sample"
        return final      # 这个 "sample" 对应 x_{t-1}, 即每一步的状态, 到最后一步会把 x_0 输出

    # 生成一系列的 x_{t-1}，从 t = T 逐渐到 t = 0
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        input_batch=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        indices = list(range(self.num_timesteps))[::-1]     # 表示的是 t 的序号，indices_t 不一定等于 t

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm  # 自动根据环境选择进度条

            indices = tqdm(indices)

        # stx()
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                # stx()
                output = self.p_sample(
                    model,
                    input_batch,                            # img = x_t, p_sample 会返回 x_{t-1}. img 的初始值是 noise
                    t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
                yield output
                # 修改 input_batch 里面的迭代字典
                # 应该在 p_sample 里面修改 input_batch 的关键字, 然后这边就直接传就好 
                input_batch = output["input_batch"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # 计算 q(x_{t-1} | x_t, x_0) - 此处的 x_0 是真实值
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )

        # 调用 model 计算 p(x_{t-1} | x_t) - 此处的 x_0 是模型猜的
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        # 计算 KL(q(x_{t-1} | x_t, x_0) || p(x_{t-1} | x_t)), true 和 out 的 KL 散度
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        # 第一步输出对数似然，后面输出 KL 散度
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, batch, t, model_kwargs=None, noise=None, mode='training', create_visual = False, x0_loss_weight = 1.0):
        """
        被调用形式: loss_metrics = self.diffusion.training_losses(self.denoiser, batch, t)
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param batch: data batch - 一个 dict: dict_keys(['image', 'c2w', 'fxfycxcy', 'index', 'depth', 'normal'])
        :param t: a batch of timestep indices.

        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
                 返回一个 loss 的 dict
        """
        # 取 batch 的第二个 view 开始作为 x_0
        # 这里的 batch 还是一整个的 batch
        # (b, v, c, h, w)
        img_condition = batch["image"][:,0:1, ...]          # [8, 1, 3, 256, 256]
        x_start = batch["image"][:,1:4, ...].clone()                 # [8, 3, 3, 256, 256]

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)                  # [8, 3, 3, 256, 256]

        # # x_0, t -> x_t
        # q(x_t | x_0) = N(x_t; sqrt(alpha_t_bar) x_0, sqrt(1 - alpha_t_bar))
        # x_t: (b, v-1, c, h, w)
        x_t = self.q_sample(x_start, t, noise=noise)


        batch['image_noisy'] = x_t          # 这个值不会引起别的改变, 因为是新开辟的 key

        # stx()

        terms = {}

        # 计算 KL(q(x_{t-1} | x_t, x_0) || p(x_{t-1} | x_t)) 散度 Loss, KL 散度 Loss 用得其实不多
        # 这里的 if 逻辑有点奇怪，就是说如果是 KL 散度 Loss 的话，就直接返回 KL 散度 Loss
        # 如果是 MSE Loss 的话，不仅要算 MSE Loss，还要算 KL 散度 Loss，返回的 loss 是两者之和
        # 此处是 False, 不执行
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps

        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:

            denoiser_output = model(batch, t, **model_kwargs, mode = mode, create_visual=create_visual)   # 过 denoiser
            model_output = denoiser_output['render_input']
           
           #######################################################################################################
           #############################  可视化 debug - 计算 loss 前把所有的图片存下来进行调试  #######################
           #######################################################################################################

            # debug_path = os.path.join("/sensei-fs/users/yuanhaoc/gslrm_dit/minLRM/debug", "gaussian_diffusion")
            # os.makedirs(debug_path, exist_ok=True)

            # im = model_output
            # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
            # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
            # Image.fromarray(im).save(os.path.join(debug_path, "model_render_input.png"))

            # im = denoiser_output['render']
            # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
            # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
            # Image.fromarray(im).save(os.path.join(debug_path, "model_render_target.png"))

            # im = denoiser_output['input']['image']
            # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
            # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
            # Image.fromarray(im).save(os.path.join(debug_path, "input_image.png"))

            # im = denoiser_output['target']['image']
            # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
            # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
            # Image.fromarray(im).save(os.path.join(debug_path, "target_image.png"))

            # im_x_start = x_start
            # im_x_start = rearrange(im_x_start, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
            # im_x_start = (im_x_start * 255).clip(0.0, 255.0).astype(np.uint8)
            # Image.fromarray(im_x_start).save(os.path.join(debug_path, "x_start.png"))

            # im = x_t
            # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
            # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
            # Image.fromarray(im).save(os.path.join(debug_path, "x_t.png"))


            # combo_image_condition = th.cat([img_condition, x_t], dim=1)
            # im = combo_image_condition
            # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
            # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
            # Image.fromarray(im).save(os.path.join(debug_path, "combo_input_image.png"))


            # stx()

            ################################################################################################################
            ############################################    可视化 debug 结束    #############################################
            ################################################################################################################


            '''
                model_output 理论上也是一个字典
                dict_keys(['input', 'target', 'gaussians', 'pixelalign_xyz', 'img_tokens', 'loss_metrics', 'render'])
                ——> dict_keys(['input', 'target', 'gaussians', 'pixelalign_xyz', 'img_tokens', 'loss_metrics', 'render', 'render_input'])
                要从中抽出 multi-view 的 x0 然后变成 x_{t-1}
            '''

            # learn_sigma = True, 所以模型不仅要 predict mean, 还要 predict variance
            # 我已经设置成 False 了
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb_loss"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb_loss"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],                                                       # x_{t-1}
                ModelMeanType.START_X: x_start,                             # x_0
                ModelMeanType.EPSILON: noise,                               # eps
            }[self.model_mean_type]  # mean type 决定了在什么上边算 MSE loss
            # stx()
            assert model_output.shape == target.shape == x_start.shape



            #######################################################################################################
            ###################################  x0_loss 会导致掉点 - 再多渲染一次看看  ###############################
            #######################################################################################################

            # debug_path = os.path.join("/sensei-fs/users/yuanhaoc/gslrm_dit/minLRM/debug_2object", "gaussian_diffusion")
            # os.makedirs(debug_path, exist_ok=True)

            # im = target
            # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
            # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
            # Image.fromarray(im).save(os.path.join(debug_path, "input_gt.png"))

            # im = model_output
            # im = rearrange(im, "b v c h w -> (b h) (v w) c").detach().cpu().numpy()
            # im = (im * 255).clip(0.0, 255.0).astype(np.uint8)
            # Image.fromarray(im).save(os.path.join(debug_path, "render_input.png"))

            # stx()

            #######################################################################################################
            ##################################################  调试结束  ##########################################
            #######################################################################################################


            eps_loss = False
            x0_loss = False
            if self.model_mean_type == ModelMeanType.START_X and eps_loss:
                predict_eps = self._predict_eps_from_xstart(x_t=x_t, t=t, pred_xstart=model_output)
                terms["mse_x0_loss"] = mean_flat((noise - predict_eps) ** 2)
            elif self.model_mean_type == ModelMeanType.EPSILON and x0_loss:
                predict_x0 = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
                terms["mse_x0_loss"] = mean_flat((x_start - predict_x0) ** 2)
            else:
                terms["mse_x0_loss"] = mean_flat((target - model_output) ** 2) * x0_loss_weight
            

            # 前面的 predict variance 已经被设置成 False
            # terms 字典里没有 vb
            output_loss_metrics = denoiser_output['loss_metrics']
            if "vb_loss" in terms:
                terms["loss"] = terms["mse_x0_loss"] + terms["vb_loss"]      
            else:
                terms["loss"] = terms["mse_x0_loss"]
            
            # 遍历 output_loss_metrics 的键值对
            for key, value in output_loss_metrics.items():
                if key == 'loss':
                    terms[key] = terms[key] + value     # term 里面的 loss 是全部的 loss 之和
                else:
                    terms[key] = value
        else:
            raise NotImplementedError(self.loss_type)

        return terms, denoiser_output


# 从一个一维的 numpy 数组中提取出一批索引对应的值，并将这些值转换为 PyTorch 的 Tensor 格式
# 按照选取的 timestep 去取预先计算好的数值
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)
