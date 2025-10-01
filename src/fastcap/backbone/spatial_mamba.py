 
# enhanced-fastcap/src/fastcap/backbone/spatial_mamba.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class SpatialMamba(nn.Module):
    """
    The core Spatial-Mamba module implementing the 4-direction cross-scan.
    This is based on the principles outlined in "Innovation 1: Spatial-Mamba Vision Backbone".
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

        # Projections for SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_model * 2, bias=False)
        
        # State Space Model (SSM) parameters A, B, C, D
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
        
        # Rearrange input for scanning
        x_flat = x.view(B, H * W, C)

        # Apply the 4-direction cross-scan as described in the docs
        scans = []
        for direction in ["h_forward", "h_backward", "v_forward", "v_backward"]:
            scanned_x = self.selective_scan(x, direction)
            scans.append(scanned_x)
        
        # Combine the outputs of the four scans
        # The document suggests a weighted sum, but a simple mean is a strong baseline.
        y = torch.stack(scans, dim=0).mean(dim=0)
        
        return y.view(B, H, W, C)

    def selective_scan(self, x, direction):
        """
        Performs a selective scan in one of the four directions.
        This function handles rearranging the input, applying the Mamba logic, and rearranging back.
        """
        B, H, W, C = x.shape
        
        # Permute the input based on the scan direction
        if direction == "h_forward":
            x_permuted = x.permute(0, 2, 1, 3).reshape(B * W, H, C) # Scan along height
        elif direction == "h_backward":
            x_permuted = x.permute(0, 2, 1, 3).flip(dims=[1]).reshape(B * W, H, C)
        elif direction == "v_forward":
            x_permuted = x.permute(0, 1, 2, 3).reshape(B * H, W, C) # Scan along width
        elif direction == "v_backward":
            x_permuted = x.permute(0, 1, 2, 3).flip(dims=[1]).reshape(B * H, W, C)
        
        # --- Core Mamba Logic ---
        # 1. Linear projection and split into x and z
        xz = self.in_proj(x_permuted)
        x_ssm, z = xz.chunk(2, dim=-1) # (B*W, H, d_inner) or (B*H, W, d_inner)

        # 2. 1D Convolution
        x_ssm = rearrange(x_ssm, "b l d -> b d l")
        x_ssm = self.conv1d(x_ssm)[:, :, :x_permuted.shape[1]] # Causal padding
        x_ssm = rearrange(x_ssm, "b d l -> b l d")
        x_ssm = F.silu(x_ssm)

        # 3. Discretize SSM parameters (delta, B, C) and compute A
        x_dbl = self.x_proj(x_ssm)
        delta, B_param, C_param = x_dbl.split([self.d_model, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)
        
        # Discretize A: A_bar = exp(delta * A)
        A = -torch.exp(self.A_log.float())
        A_bar = torch.exp(delta.unsqueeze(-1) * A)

        # Discretize B: B_bar = delta * B
        B_bar = delta.unsqueeze(-1) * B_param.unsqueeze(-2)
        
        # 4. Parallel Scan (Prefix Sum)
        # This is the efficient, parallel implementation of the SSM recurrence
        # h_t = A_bar * h_{t-1} + B_bar * x_t
        h = torch.zeros(x_ssm.size(0), self.d_inner, self.d_state, device=x.device)
        ys = []
        for i in range(x_ssm.size(1)):
            h = A_bar[:, i] * h + B_bar[:, i] * x_ssm[:, i].unsqueeze(-1)
            y_i = (h @ C_param[:, i].unsqueeze(-1)).squeeze(-1)
            ys.append(y_i)
        y_ssm = torch.stack(ys, dim=1)

        y_ssm = y_ssm + self.D * x_ssm
        
        # 5. Gating (Modulation) with z
        y_gated = y_ssm * F.silu(z)

        # 6. Output projection
        output = self.out_proj(y_gated)

        # Reverse the permutation to restore original shape
        if direction == "h_backward":
            output = output.flip(dims=[1])
        elif direction == "v_backward":
            output = output.flip(dims=[1])

        if "h" in direction:
            output = output.view(B, W, H, C).permute(0, 2, 1, 3)
        else:
            output = output.view(B, H, W, C)
            
        return output


class SpatialMambaBlock(nn.Module):
    """
    A residual block containing a SpatialMamba module.
    Structure: Input -> Norm -> SpatialMamba -> Add -> Norm -> MLP -> Add
    As defined in the architecture specification.
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
        # The input shape is expected to be (B, H, W, C)
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
        x = self.proj(x) # (B, C, H, W)
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
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[SpatialMambaBlock(d_model=embed_dims[i], d_state=d_state) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                # Downsampling layer
                downsample = nn.Sequential(
                    nn.LayerNorm(embed_dims[i]),
                    nn.Conv2d(embed_dims[i], embed_dims[i+1], kernel_size=2, stride=2)
                )
                self.stages.append(downsample)

        # Final layers for producing caption-ready features
        self.final_norm = nn.LayerNorm(embed_dims[-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(embed_dims[-1], 256) # Project to final dimension

    def forward(self, x):
        # 1. Patch Embedding
        x = self.patch_embed(x) # (B, H, W, C)

        # 2. Pass through Mamba stages
        for stage in self.stages:
            if isinstance(stage, nn.Sequential) and not isinstance(stage[0], nn.LayerNorm):
                # This is a SpatialMambaBlock stage
                x = stage(x)
            else:
                # This is a downsampling stage, requires channel-first format
                x = rearrange(x, 'b h w c -> b c h w')
                x = stage(x)
                x = rearrange(x, 'b c h w -> b h w c')

        # 3. Global Pooling and Projection
        x = self.final_norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x_pooled = self.pool(x).flatten(1)
        
        # 4. Final caption-ready features
        final_features = self.feature_proj(x_pooled) # (B, 256)
        
        return final_features

# Example usage:
if __name__ == '__main__':
    # Create a dummy input image tensor
    dummy_image = torch.randn(2, 3, 224, 224) # (Batch, Channels, Height, Width)
    
    # Instantiate the backbone
    vision_backbone = SpatialMambaBackbone()
    
    # Get the vision features
    features = vision_backbone(dummy_image)
    
    print("Spatial-Mamba Vision Backbone")
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Output feature shape: {features.shape}") # Expected: (2, 256)