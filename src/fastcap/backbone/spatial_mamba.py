# In src/fastcap/backbone/spatial_mamba.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return x

class DownsampleLayer(nn.Module):
    """
    Downsamples the feature map.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        if dt_rank == "auto":
            dt_rank = (self.d_inner // 16)
        
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
    def ssm(self, x):
        (B, L, C) = x.shape
        D = self.d_inner
        N = self.d_state

        x_and_z = self.in_proj(x)
        x_ssm, z = x_and_z.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))
        x_conv = x_conv[:, :, :L].transpose(1, 2)
        x_ssm = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_ssm)
        dt, B_param, C_param = torch.split(x_dbl, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        
        delta = self.dt_proj(dt)
        delta = F.softplus(delta)
        A = -torch.exp(self.A_log.float())

        # --- CORRECTED EINSUM LOGIC ---
        # Discretize A and B, performing the operation in a clear, correct manner
        delta_A = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        
        # This was the problematic line. We replace it with a clearer equivalent.
        delta_x_prod = delta.unsqueeze(-1) * x_ssm.unsqueeze(-1) # (B, L, D, 1)
        B_param_prod = B_param.unsqueeze(2) # (B, L, 1, N)
        delta_B_x = delta_x_prod * B_param_prod # (B, L, D, N) via broadcasting

        # Parallel scan
        h = torch.zeros(B, D, N, device=x.device, dtype=torch.float32)
        y_ssm = torch.zeros_like(x_ssm)
        
        for i in range(L):
            h = delta_A[:, i] * h + delta_B_x[:, i]
            y_ssm[:, i] = torch.einsum('bdn,bn->bd', h, C_param[:, i])

        y_ssm = y_ssm + self.D.unsqueeze(0).unsqueeze(0) * x_ssm
        y_gated = y_ssm * F.silu(z)
        output = self.out_proj(y_gated)
        return output
        
    def forward(self, x):
        B, H, W, C = x.shape
        x_seq = rearrange(x, 'b h w c -> b (h w) c')
        
        x_fwd = self.ssm(x_seq)
        x_bwd = self.ssm(x_seq.flip(dims=[1])).flip(dims=[1])
        
        x_v_fwd = self.ssm(rearrange(x_seq, 'b (h w) c -> b (w h) c', h=H, w=W))
        x_v_fwd = rearrange(x_v_fwd, 'b (w h) c -> b (h w) c', h=H, w=W)
        
        x_v_bwd = self.ssm(rearrange(x_seq, 'b (h w) c -> b (w h) c', h=H, w=W).flip(dims=[1])).flip(dims=[1])
        x_v_bwd = rearrange(x_v_bwd, 'b (w h) c -> b (h w) c', h=H, w=W)

        x_out = x_fwd + x_bwd + x_v_fwd + x_v_bwd
        x_out = rearrange(x_out, 'b (h w) c -> b h w c', h=H, w=W)

        return x_out


class SpatialMambaStage(nn.Module):
    def __init__(self, d_model, d_inner, **kwargs):
        super().__init__()
        self.mixer = MambaBlock(d_model=d_model, d_inner=d_inner, **kwargs)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mixer(self.norm(x))
        return x


class SpatialMambaBackbone(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, depths=[2, 2, 9, 2], d_inner_mult=2):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dim=embed_dim)
        
        self.stages = nn.ModuleList()
        current_dim = embed_dim
        
        for i, depth in enumerate(depths):
            stage_modules = []
            if i > 0:
                stage_modules.append(DownsampleLayer(current_dim, current_dim * 2))
                current_dim *= 2
            
            for _ in range(depth):
                stage_modules.append(SpatialMambaStage(
                    d_model=current_dim, 
                    d_inner=current_dim * d_inner_mult
                ))
            
            self.stages.append(nn.Sequential(*stage_modules))
            
        self.final_norm = nn.LayerNorm(current_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_norm(x)
        
        B, H, W, C = x.shape
        x_seq = x.view(B, H * W, C)
        return x_seq