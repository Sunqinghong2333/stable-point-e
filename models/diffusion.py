import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import math
import torch.nn as nn

from .common import *
from .checkpoint import checkpoint
from .gaussian_diffusion import GaussianDiffusion, mean_flat, get_named_beta_schedule


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class PointwiseNet(Module):

    def __init__(self, token_cond=True, time_token_cond=True, device="cuda", dtype=torch.float32, width=512, init_scale=0.25, layers=12, heads=8, cond_drop_prob=0.1, context_dim=768, n_ctx=1024, input_channels=3, output_channels=6):
        super().__init__()
        self.token_cond = token_cond
        self.time_token_cond = time_token_cond
        self.cond_drop_prob = cond_drop_prob
        self.device = device
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.clip_embed = nn.Linear(
            context_dim, width, device=device, dtype=dtype
        )
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def _forward_with_cond(self, x, cond_as_token):

        h = self.input_proj(x.permute(0, 2, 1))
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]

        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)

    def forward(self, x, t, clip_out=None, embeddings=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            clip_out:  Shape latents. (B, F).
        """
        t_embed = self.time_embed(timestep_embedding(t.to(self.device), self.backbone.width))

        if clip_out:
            if self.training:
                mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
                clip_out = clip_out * mask[:, None].to(clip_out)

            clip_out = math.sqrt(clip_out.shape[1]) * clip_out
        else:
            clip_out = embeddings

        clip_embed = self.clip_embed(clip_out)

        cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]

        return self._forward_with_cond(x, cond)


@torch.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        yield {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "pred_xstart": denoised}
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    yield {"x": x, "pred_xstart": denoised}


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


class GaussianToKarrasDenoiser:
    def __init__(self, model, diffusion):
        from scipy import interpolate

        self.model = model
        self.diffusion = diffusion
        self.alpha_cumprod_to_t = interpolate.interp1d(
            diffusion.alphas_cumprod, np.arange(0, diffusion.num_timesteps)
        )

    def sigma_to_t(self, sigma):
        alpha_cumprod = 1.0 / (sigma**2 + 1)
        if alpha_cumprod > self.diffusion.alphas_cumprod[0]:
            return 0
        elif alpha_cumprod <= self.diffusion.alphas_cumprod[-1]:
            return self.diffusion.num_timesteps - 1
        else:
            return float(self.alpha_cumprod_to_t(alpha_cumprod))

    def denoise(self, x_t, sigmas, clip_denoised=True, model_kwargs=None):
        t = torch.tensor(
            [self.sigma_to_t(sigma) for sigma in sigmas.cpu().numpy()],
            dtype=torch.long,
            device=sigmas.device,
        )
        c_in = append_dims(1.0 / (sigmas**2 + 1) ** 0.5, x_t.ndim)
        
        out = self.diffusion.p_mean_variance(
            self.model, x_t * c_in, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        return None, out["pred_xstart"]

    def training_losses(self, x_start, sigmas, clip_denoised=True, model_kwargs=None, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        model_output, denoised = self.denoise(x_t, sigmas, clip_denoised=clip_denoised, model_kwargs=model_kwargs)

        loss = mean_flat((denoised - x_start) ** 2)

        return loss


def diffusion_from_config(schedule="cosine", steps=1024, model_mean_type="epsilon",
    model_var_type="learned_range", loss_type="mse", channel_scales=[2.0, 2.0, 2.0],
    channel_biases=[0.0, 0.0, 0.0]):
    betas = get_named_beta_schedule(schedule, steps)
    if channel_scales is not None:
        channel_scales = np.array(channel_scales)
    if channel_biases is not None:
        channel_biases = np.array(channel_biases)
    kwargs = dict(
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        channel_scales=channel_scales,
        channel_biases=channel_biases,
    )
    return GaussianDiffusion(**kwargs)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


class KarrasPoint(Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def karras_sample_progressive(self, num_points, context, texts, point_dim=3, device="cuda",
        steps=64, sigma_min=0.001, sigma_max=120, rho=7.0, 
        s_churn=3, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
        clip_denoised=True, guidance_scale=3.0, progress=False):


        batch_size = len(texts)
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)
        x_T = torch.randn([batch_size, point_dim, num_points]).to(device) * sigma_max
        sample_fn = sample_heun
        sampler_args = dict(s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise)


        diffusion = diffusion_from_config()
        model = GaussianToKarrasDenoiser(self.net, diffusion)

        stage_model_kwargs = {'texts': texts}


        def denoiser(x_t, sigma):
            _, denoised = model.denoise(
                x_t, sigma, clip_denoised=clip_denoised, model_kwargs=dict(clip_out=None, embeddings=torch.cat([context, torch.zeros_like(context)], dim=0))
            )
            return denoised

        if guidance_scale != 0 and guidance_scale != 1:
            def guided_denoiser(x_t, sigma):
                x_t = torch.cat([x_t, x_t], dim=0)
                sigma = torch.cat([sigma, sigma], dim=0)
                x_0 = denoiser(x_t, sigma)
                cond_x_0, uncond_x_0 = torch.split(x_0, len(x_0) // 2, dim=0)
                x_0 = uncond_x_0 + guidance_scale * (cond_x_0 - uncond_x_0)
                return x_0
        else:
            guided_denoiser = denoiser

        for obj in sample_fn(
            guided_denoiser,
            x_T,
            sigmas,
            progress=progress,
            **sampler_args,
        ):
            yield diffusion.unscale_out_dict(obj)


    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """


    def sample(self, num_points, context, texts, point_dim=3, ret_traj=False):

        samples_it = self.karras_sample_progressive(num_points, context, texts, point_dim)
        batch_size = context.size(0)
        for x in samples_it:
            samples = x["pred_xstart"][:batch_size]
            yield samples



       