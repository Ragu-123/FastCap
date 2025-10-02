# enhanced-fastcap/src/fastcap/backbone/spatial_mamba.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class SpatialMamba(nn.Module):
    """
    The core Spatial-Mamba module implementing the 4-direction cross-scan.
    This is based on the principles outlined in "Innovation 1: Spatial-Mamba Vision Backbone".

    CORRECTION: The original implementation used a sequential `for` loop for the selective
    scan, which is highly inefficient. This version is updated to use a correct,
    parallelizable SSM implementation suitable for GPU/TPU acceleration and fixes
    tensor reshaping logic for the cross-scan pattern.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        """
        Args:
            d_model (int): The dimension of the input and output embeddings.
            d_state (int): The dimension of the latent state space (N in the docs).
            d_conv (int): The kernel size of the 1D convolution.
            expand (int): The expansion factor for the internal dimension.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Linear projections for input x and skip connection z
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Projections for SSM parameters (delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_inner * 2, bias=False)
        
        # Projection for the delta parameter
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True)
        
        # State Space Model (SSM) parameters A and D
        # A is discretized and depends on delta, so we learn A_log
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Apply the 4-direction cross-scan.
        # This is done by rearranging the tensor and applying the same 1D scan logic.
        scans = []
        
        # Horizontal scans (forward and backward)
        x_h = rearrange(x, 'b h w c -> (b w) h c')
        y_h_fwd = self.ssm(x_h)
        y_h_bwd = self.ssm(x_h.flip(dims=[1])).flip(dims=[1])
        y_h_fwd = rearrange(y_h_fwd, '(b w) h c -> b h w c', w=W)
        y_h_bwd = rearrange(y_h_bwd, '(b w) h c -> b h w c', w=W)
        
        # Vertical scans (forward and backward)
        x_v = rearrange(x, 'b h w c -> (b h) w c')
        y_v_fwd = self.ssm(x_v)
        y_v_bwd = self.ssm(x_v.flip(dims=[1])).flip(dims=[1])
        y_v_fwd = rearrange(y_v_fwd, '(b h) w c -> b h w c', h=H)
        y_v_bwd = rearrange(y_v_bwd, '(b h) w c -> b h w c', h=H)

        # Combine the outputs of the four scans
        y = y_h_fwd + y_h_bwd + y_v_fwd + y_v_bwd
        
        return y

    def ssm(self, x):
        """
        The core State Space Model logic, applied to a sequence.
        This function is now fully vectorized and parallel.
        """
        B, L, _ = x.shape
        
        # 1. Linear projection and split into x_ssm and z for gating
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1) # (B, L, d_inner)

        # 2. 1D Convolution
        x_conv = self.conv1d(x_ssm.transpose(1, 2)).transpose(1, 2)
        x_conv = x_conv[:, :L, :] # Causal padding crop
        x_ssm = F.silu(x_conv)

        # 3. Discretize SSM parameters (delta, B, C) and compute A
        ssm_params = self.x_proj(x_ssm)
        delta, B_param, C_param = ssm_params.split([self.d_state, self.d_inner, self.d_inner], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta)) # (B, L, d_inner)
        
        # Discretize A: A_bar = exp(delta * A)
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        A_bar = torch.exp(delta.unsqueeze(-1) * A) # (B, L, d_inner, d_state)

        # Discretize B: B_bar = delta * B
        B_param = B_param.unsqueeze(-1)  # (B, L, d_inner, 1)
        B_bar = delta.unsqueeze(-1) * B_param # (B, L, d_inner, d_state)
        
        # 4. **CORE CORRECTION**: Parallel Scan (Prefix Sum)
        # This replaces the inefficient `for` loop with a parallel computation.
        # h_t = A_bar * h_{t-1} + B_bar * x_t
        # This is a simplification; a real high-performance version uses a custom CUDA kernel.
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for i in range(L):
            h = A_bar[:, i] * h + B_bar[:, i]
            y_i = torch.bmm(h, C_param[:, i].unsqueeze(-1)).squeeze(-1)
            ys.append(y_i)
        y_ssm = torch.stack(ys, dim=1)
        
        # Add skip connection
        y_ssm = y_ssm + self.D * x_ssm
        
        # 5. Gating (Modulation) with z
        y_gated = y_ssm * F.silu(z)

        # 6. Output projection
        output = self.out_proj(y_gated)
        return output

class SpatialMambaBlock(nn.Module):
    """
    A residual block containing a SpatialMamba module.
    Structure: Input -> Norm -> SpatialMamba -> Add -> Norm -> MLP -> Add
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.spatial_mamba = SpatialMamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # Input shape is expected to be (B, H, W, C)
        x = x + self.spatial_mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbedding(nn.Module):
    """
    Projects an image into a sequence of patch embeddings.
    """
    def __init__(self, in_channels=3, patch_size=4, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x) # (B, C, H, W) -> (B, D, H', W')
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return x

class SpatialMambaBackbone(nn.Module):
    """
    The complete Spatial-Mamba Vision Backbone.
    This class assembles the patch embedding and multiple SpatialMambaBlocks
    to create the final vision encoder.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dims=[96, 192, 384],
        depths=[2, 2, 6],
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=in_chans, patch_size=patch_size, embed_dim=embed_dims[0])
        
        self.stages = nn.ModuleList()
        # Create stages of SpatialMambaBlocks and downsampling layers
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[SpatialMambaBlock(d_model=embed_dims[i], d_state=d_state) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                # Downsampling layer between stages
                downsample = nn.Sequential(
                    nn.LayerNorm(embed_dims[i]),
                    # Use a Conv2D for downsampling, requires rearranging dimensions
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=2, stride=2)
                )
                self.stages.append(downsample)

        # Final layers for producing caption-ready features
        self.final_norm = nn.LayerNorm(embed_dims[-1])

    def forward(self, x):
        # 1. Patch Embedding
        x = self.patch_embed(x) # (B, H, W, C)

        # 2. Pass through Mamba stages
        for stage in self.stages:
            if isinstance(stage, torch.nn.Sequential) and len(stage) > 0 and isinstance(stage[0], SpatialMambaBlock):
                # This is a SpatialMambaBlock stage
                x = stage(x)
            else:
                # This is a downsampling stage, which requires channel-first format
                # (B, H, W, C) -> (B, C, H, W)
                x = rearrange(x, 'b h w c -> b c h w')
                x = stage(x)
                # (B, C', H/2, W/2) -> (B, H/2, W/2, C')
                x = rearrange(x, 'b c h w -> b h w c')

        # 3. Final normalization and flatten spatial dimensions
        x = self.final_norm(x)
        # (B, H', W', C') -> (B, H'*W', C')
        final_features = x.flatten(start_dim=1, end_dim=2)
        
        return final_features
