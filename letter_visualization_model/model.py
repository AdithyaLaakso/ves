import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings
from torch.utils.checkpoint import checkpoint

class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads

        # Number of possible relative positions = (2*Wh-1) * (2*Ww-1)
        num_rel_positions = (2*window_size[0]-1) * (2*window_size[1]-1)

        self.relative_bias_table = nn.Parameter(
            torch.zeros(num_rel_positions, num_heads)
        )

        # Precompute pairwise relative positions
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)

        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2*window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)

    @torch.compile
    def forward(self):
        # (num_rel_positions, num_heads) -> (Wh*Ww, Wh*Ww, num_heads)
        bias = self.relative_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size[0]*self.window_size[1],
                         self.window_size[0]*self.window_size[1], -1)  # (N, N, num_heads)
        return bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)

class Attention(nn.Module):
    def __init__(self,
                dim,
                num_heads: int=1,
                qkv_bias: bool=False,
                qk_scale= None):

        super().__init__()

        ## Define Constants
        self.num_heads = num_heads
        self.head_dim = dim[0] // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        ## Define Layers
        self.qkv = nn.Linear(dim[0], dim[0] * 3, bias=qkv_bias, device=settings.device)
        #### Each token gets projected from starting length (dim) to channel length (chan) 3 times (for each Q, K, V)
        self.proj = nn.Linear(dim[0], dim[0])

        self.relative_position_bias = RelativePositionBias((settings.patch_size, settings.patch_size), self.num_heads)

        self.bias = nn.Parameter(torch.randn(1024, 1024))

    @torch.compile
    def forward(self, x):
        B, N, C = x.shape
        # x: (batch, num_tokens, chan)

        # --- Calculate Q, K, V ---
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        # qkv: (3, batch, heads, num_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- Attention scores ---
        dk = self.head_dim
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)  # (B, heads, N, N)

        # --- Add relative position bias ---
        bias = self.relative_position_bias()  # (heads, N, N)
        bias = bias.to(dtype=attn_scores.dtype, device=attn_scores.device)
        attn_scores = attn_scores + self.bias.unsqueeze(0)  # (B, heads, N, N)

        # --- Attention weights ---
        attn = attn_scores.softmax(dim=-1)

        # --- Apply attention to values ---
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # --- Output projection ---
        out = self.proj(out)  # (B, N, C)

        # --- Residual connection ---
        out = x + out  # (B, N, C)

        return out

@torch.compile
def get_2d_sinusoidal_encoding(n_h, n_w, d_model):
    """
    n_h: number of patches along height
    n_w: number of patches along width
    d_model: embedding dimension (must be even)
    Returns: (n_h*n_w, d_model)
    """
    assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sinusoidal encoding"

    pe = torch.zeros(n_h, n_w, d_model)

    # Split embedding dimension across row and col encodings
    d_model_half = d_model // 2
    d_model_quarter = d_model // 4

    div_term_row = torch.exp(torch.arange(0, d_model_quarter, 2) * -(math.log(10000.0) / d_model_quarter))
    div_term_col = torch.exp(torch.arange(0, d_model_quarter, 2) * -(math.log(10000.0) / d_model_quarter))

    pos_row = torch.arange(n_h).unsqueeze(1)  # (n_h, 1)
    pos_col = torch.arange(n_w).unsqueeze(1)  # (n_w, 1)

    # Row encoding
    pe[:, :, 0:d_model_quarter:2] = torch.sin(pos_row * div_term_row).unsqueeze(1)
    pe[:, :, 1:d_model_quarter:2] = torch.cos(pos_row * div_term_row).unsqueeze(1)

    # Col encoding
    pe[:, :, d_model_quarter:d_model_half:2] = torch.sin(pos_col * div_term_col).unsqueeze(0)
    pe[:, :, d_model_quarter+1:d_model_half:2] = torch.cos(pos_col * div_term_col).unsqueeze(0)

    return pe.view(-1, d_model)  # (num_patches, d_model)

class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_size,
                              kernel_size=patch_size, stride=patch_size)

        self.n_h = image_size // patch_size
        self.n_w = image_size // patch_size
        self.embed_size = embed_size

        # Fixed 2D sinusoidal encodings
        pe = get_2d_sinusoidal_encoding(self.n_h, self.n_w, embed_size)
        self.register_buffer("positional_encoding", pe, persistent=False)

    @torch.compile
    def forward(self, x):
        # B = x.shape[0]
        x = self.proj(x)               # (B, embed_size, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_size)
        x = x + self.positional_encoding.unsqueeze(0)
        return x

# Memory-optimized MultiLayerPerceptron with optional gradient checkpointing
class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size, dropout, use_gelu=True, expansion_factor=4):
        super().__init__()
        # Reduced expansion factor from 4x to configurable (default 4, can reduce to 2-3)
        hidden_size = embed_size * expansion_factor

        # Use GELU instead of Sigmoid for better performance and memory efficiency
        activation = nn.GELU() if use_gelu else nn.Sigmoid()

        self.spatial_mix = nn.Linear(embed_size, hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            activation,
            nn.Linear(hidden_size, embed_size),
            # nn.LayerNorm(embed_size),
            nn.Dropout(p=dropout),
        )

        self.embed_size = embed_size
        self.hidden_size = hidden_size

    @torch.compile
    def forward(self, x):
        # x_spatial = self.spatial_mix(x)
        # x_channel = self.layers(x_spatial)
        # return x + x_channel
        x_channel = self.layers(x)
        return x_channel

# memory-optimized self-attention encoder block with gradient checkpointing option
class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, use_checkpoint=False):
        super().__init__()
        self.embed_size = embed_size
        self.use_checkpoint = use_checkpoint
        self.ln1 = nn.LayerNorm(embed_size)

        # use scaled_dot_product_attention for memory efficiency (pytorch 2.0+)
        self.mha = Attention(
            dim=(embed_size, embed_size),
            num_heads=num_heads,
            qkv_bias=False,
            qk_scale=None
        )

        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MultiLayerPerceptron(embed_size, dropout, expansion_factor=2)

    @torch.compile
    def _attention_forward(self, x):
        y = self.ln1(x)
        return x + self.mha(y)

    @torch.compile
    def _mlp_forward(self, x):
        return x + self.mlp(self.ln2(x))

    @torch.compile
    def forward(self, x):
        # use gradient checkpointing to trade compute for memory
        if self.use_checkpoint and self.training:
            x = checkpoint(self._attention_forward, x, use_reentrant=False)
            x = checkpoint(self._mlp_forward, x, use_reentrant=False)
        else:
            x = self._attention_forward(x)
            x = self._mlp_forward(x)
        return x

# Memory-optimized output projection with progressive upsampling
class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, output_dims, output_size, use_progressive_upsampling=False):
        super().__init__()
        self.patch_size = patch_size
        self.output_dims = output_dims
        self.output_size = output_size
        self.use_progressive_upsampling = use_progressive_upsampling


        self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)
        self.att_gate = AttentionGate(F_g=output_dims, F_l=settings.in_channels)

        # Option 1: Direct projection (memory intensive)
        # self.projection = nn.Linear(embed_size, patch_size * patch_size * output_dims)

        # Option 2: Progressive upsampling (memory efficient)
        # Use smaller intermediate projection
        intermediate_dims = min(embed_size // 2, 128)
        self.projection = nn.Sequential(
            nn.Linear(embed_size, intermediate_dims),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dims, patch_size * patch_size * output_dims)
        )

    @torch.compile
    def forward(self, x, skip):
        x = self.projection(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fold(x)

        skip_gated = self.att_gate(x, skip)
        x = x * skip_gated

        x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)

        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int=1):
        super().__init__()
        # Gating from decoder (low-res feature)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        # Skip from encoder (high-res feature)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        # Psi outputs attention mask
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: decoder feature (from projection path)
        # x: skip feature (original input / encoder feature)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class InputNormalization(nn.Module):
    """Normalize input from 0-255 range to 0-1 range"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Handle both uint8 and float inputs
        if x.dtype == torch.uint8:
            return x.float() / 255.0
        elif x.max() > 1.0:
            return x / 255.0
        return x

class VisionTransformerForSegmentation(nn.Module):
    def __init__(self, use_gradient_checkpointing=True):
        super().__init__()
        self.output_size = settings.output_size
        self.in_channels = settings.in_channels
        self.out_channels = settings.out_channels
        self.image_size = settings.image_size
        self.patch_size = settings.patch_size
        self.embed_size = settings.embed_size
        self.num_blocks = settings.num_blocks
        self.num_heads = settings.num_heads
        self.dropout = settings.dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Create encoder blocks with optional gradient checkpointing
        heads = [
            SelfAttentionEncoderBlock(
                self.embed_size,
                self.num_heads,
                self.dropout,
                use_checkpoint=use_gradient_checkpointing
            )
            for _ in range(self.num_blocks)
        ]

        self.vit_input = VisionTransformerInput(
            self.image_size, self.patch_size, self.in_channels, self.embed_size
        )
        self.encoder_blocks = nn.ModuleList(heads)
        self.output_projection = OutputProjection(
            self.image_size, self.patch_size, self.embed_size,
            self.out_channels, self.output_size,
            use_progressive_upsampling=True
        )

    @torch.compile
    def forward(self, x):
        # Input processing
        input = x.clone()
        x = self.vit_input(x)

        # Process through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        # Output projection
        x = self.output_projection(x, input)
        return x

# Usage example with memory monitoring
def build_model():
    # Create model with memory optimizations
    model = VisionTransformerForSegmentation( use_gradient_checkpointing=True)

    if settings.load_from is not None:
        model.load_state_dict(settings.load_from)

    return model
