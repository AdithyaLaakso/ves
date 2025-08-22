import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings
from torch.utils.checkpoint import checkpoint
def tokens_to_map(x, n_h, n_w):
    # x: (B, N, C) -> (B, C, H, W)
    B, N, C = x.shape
    assert N == n_h * n_w, "N must equal n_h*n_w"
    return x.transpose(1, 2).reshape(B, C, n_h, n_w)

def map_to_tokens(x):
    # x: (B, C, H, W) -> (B, H*W, C)
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2)

class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        num_rel_positions = (2*window_size[0]-1) * (2*window_size[1]-1)
        self.relative_bias_table = nn.Parameter(torch.zeros(num_rel_positions, num_heads))

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

    def forward(self):
        bias = self.relative_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size[0]*self.window_size[1],
                         self.window_size[0]*self.window_size[1], -1)
        return bias.permute(2, 0, 1).contiguous()  # (heads, N, N)

class Attention(nn.Module):
    def __init__(self, dim, num_heads: int=1, qkv_bias: bool=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim[0] // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim[0], dim[0] * 3, bias=qkv_bias, device=settings.device)
        self.proj = nn.Linear(dim[0], dim[0])
        # RPB will be set by the caller per scale (since window sizes differ)
        self.relative_position_bias = None
        # keep your (global) learnable bias but match shape on the fly
        self.global_bias = None  # lazy init

    def _ensure_bias(self, B, heads, N, device, dtype):
        if (self.global_bias is None) or (self.global_bias.shape[0] < N) or (self.global_bias.shape[1] < N):
            size = max(N, 1024)  # avoid frequent re-allocs; cap can be larger if you like
            self.global_bias = nn.Parameter(torch.zeros(size, size, device=device, dtype=dtype))
        return self.global_bias[:N, :N]

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        if self.relative_position_bias is not None:
            rpb = self.relative_position_bias().to(dtype=attn_scores.dtype, device=attn_scores.device)
            attn_scores = attn_scores + rpb.unsqueeze(0)

        gb = self._ensure_bias(B, self.num_heads, N, attn_scores.device, attn_scores.dtype)
        attn_scores = attn_scores + gb.unsqueeze(0)

        attn = attn_scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

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
        pe = get_2d_sinusoidal_encoding(self.n_h, self.n_w, embed_size)
        self.register_buffer("positional_encoding", pe, persistent=False)

    @torch.compile
    def forward(self, x):
        x = self.proj(x)                       # (B, embed, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)       # (B, N, embed)
        x = x + self.positional_encoding.unsqueeze(0)
        return x, (self.n_h, self.n_w)

class MultiScaleVisionTransformerInput(nn.Module):
    def __init__(self, image_size, in_channels, embed_size, patch_sizes):
        super().__init__()
        self.scales = nn.ModuleDict({
            str(p): VisionTransformerInput(image_size, p, in_channels, embed_size)
            for p in patch_sizes
        })
        self.patch_sizes = patch_sizes

    def forward(self, x):
        # returns dict: {p: (tokens, (n_h, n_w))}
        return {p: self.scales[str(p)](x) for p in self.patch_sizes}

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

class ConvDilatedMLP(nn.Module):
    """
    Token MLP enhanced with convs:
      tokens -> map -> [1x1 -> DW(dilated) -> 1x1] -> tokens
    """
    def __init__(self, embed_size, dropout=0.0, hidden_ratio=2, dilation=2):
        super().__init__()
        hidden = embed_size * hidden_ratio
        self.pw1 = nn.Conv2d(embed_size, hidden, kernel_size=1)
        self.dw_dil = nn.Conv2d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation, groups=hidden)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, embed_size, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, grid_size):
        n_h, n_w = grid_size
        fmap = tokens_to_map(x, n_h, n_w)          # (B, C, H, W)
        y = self.pw1(fmap)
        y = self.act(y)
        y = self.dw_dil(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.drop(y)
        return map_to_tokens(y)                    # (B, N, C)

class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, window_size, use_checkpoint=False, mlp_dilation=2):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.ln1 = nn.LayerNorm(embed_size)
        self.attn = Attention(dim=(embed_size, embed_size), num_heads=num_heads, qkv_bias=False, qk_scale=None)
        self.attn.relative_position_bias = RelativePositionBias(window_size, num_heads)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = ConvDilatedMLP(embed_size, dropout=dropout, hidden_ratio=2, dilation=mlp_dilation)

    def _attn_forward(self, x):
        y = self.ln1(x)
        return x + self.attn(y)

    def _mlp_forward(self, x, grid_size):
        return x + self.mlp(self.ln2(x), grid_size)

    @torch.compile
    def forward(self, x, grid_size):
        # if self.use_checkpoint and self.training:
        #     x = checkpoint(self._attn_forward(), x, use_reentrant=True)
        #     x = checkpoint(self._mlp_forward(), x, use_reentrant=True)
        # else:
        x = self._attn_forward(x)
        x = self._mlp_forward(x, grid_size)
        return x

class FPNHead(nn.Module):
    def __init__(self, embed_size, fpn_dim=128, scales=(8,16,32)):
        super().__init__()
        # lateral 1x1 on each scale -> fpn_dim
        self.laterals = nn.ModuleDict({str(p): nn.Conv2d(embed_size, fpn_dim, 1) for p in scales})
        self.smooth = nn.ModuleDict({str(p): nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for p in scales})
        self.scales = sorted(scales)  # small -> large (e.g., 8 < 16 < 32)

    def forward(self, feature_maps):
        """
        feature_maps: dict {p: fmap (B, C, H_p, W_p)} for p in scales
        Returns top-down fused map at the highest resolution (smallest patch size).
        """
        # top-down: start from coarsest
        ps = self.scales
        feats = {str(p): self.laterals[str(p)](feature_maps[p]) for p in ps}
        for i in reversed(range(1, len(ps))):  # from coarse -> fine
            p_coarse, p_fine = ps[i], ps[i-1]
            up = F.interpolate(feats[str(p_coarse)], size=feats[str(p_fine)].shape[-2:], mode='nearest')
            feats[str(p_fine)] = feats[str(p_fine)] + up

        # smooth convs
        for p in ps:
            feats[str(p)] = self.smooth[str(p)](feats[str(p)])

        # return finest (smallest patch) map
        return feats[str(ps[0])]

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
            nn.ReLU(inplace=False),
            nn.Linear(intermediate_dims, patch_size * patch_size * output_dims)
        )

    def forward(self, x, skip):
        x = self.projection(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fold(x)

        skip_gated = self.att_gate(x, skip)
        x = x * skip_gated

        x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)

        return x

class MultiScaleOutputHead(nn.Module):
    def __init__(self, in_dim, out_channels, output_size, use_att_gate=True, skip_channels=None):
        super().__init__()
        self.use_att_gate = use_att_gate and (skip_channels is not None)
        if self.use_att_gate:
            self.att_gate = AttentionGate(F_g=in_dim, F_l=skip_channels)

        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim, out_channels, 1)
        )
        self.output_size = output_size

    def forward(self, x, skip=None):
        if self.use_att_gate and skip is not None:
            gated = self.att_gate(x, skip)
            x = x * gated
        x = self.proj(x)
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
        self.relu = nn.ReLU(inplace=False)

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

class VisionTransformerForSegmentationMultiScale(nn.Module):
    """
    - Parallel streams for multiple patch sizes
    - Encoder blocks with Attention + ConvDilatedMLP
    - FPN to fuse per-scale features
    """
    def __init__(self, use_gradient_checkpointing=False):
        super().__init__()
        # read settings with sensible fallbacks
        self.output_size = getattr(settings, "output_size", settings.image_size)
        self.in_channels = settings.in_channels
        self.out_channels = settings.out_channels
        self.image_size = settings.image_size
        self.embed_size = settings.embed_size
        self.num_blocks = settings.num_blocks
        self.num_heads = settings.num_heads
        self.dropout = settings.dropout
        self.patch_sizes = getattr(settings, "patch_sizes", [settings.patch_size])  # e.g., [8,16,32]
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # inputs
        self.vit_inputs = MultiScaleVisionTransformerInput(
            self.image_size, self.in_channels, self.embed_size, self.patch_sizes
        )

        # per-scale encoder stacks
        self.encoders = nn.ModuleDict()
        for p in self.patch_sizes:
            n_h = self.image_size // p
            n_w = self.image_size // p
            window_size = (n_h, n_w)  # full window RPB per scale
            blocks = nn.ModuleList([
                SelfAttentionEncoderBlock(self.embed_size, self.num_heads, self.dropout,
                                          window_size=window_size,
                                          use_checkpoint=use_gradient_checkpointing,
                                          mlp_dilation=2 if i % 2 == 0 else 3)  # mix dilations
                for i in range(self.num_blocks)
            ])
            self.encoders[str(p)] = blocks

        # folding tokens back to maps per scale
        self.fold_layers = nn.ModuleDict({
            str(p): nn.Fold(output_size=(self.image_size // p, self.image_size // p),
                            kernel_size=1, stride=1)  # simple identity fold via (1x1) patches
            for p in self.patch_sizes
        })

        # FPN fusion
        self.fpn = FPNHead(embed_size=self.embed_size, fpn_dim=128, scales=self.patch_sizes)

        # output head
        self.head = MultiScaleOutputHead(
            in_dim=128,
            out_channels=self.out_channels,
            output_size=self.output_size,
            use_att_gate=True,
            skip_channels=self.in_channels
        )

    def forward(self, x):
        B, _, H, W = x.shape
        assert H == self.image_size and W == self.image_size, "resize inputs to image_size"

        # keep original for skip
        skip = x.clone()

        # 1) tokens at each scale
        per_scale = self.vit_inputs(x)  # {p: (tokens, (n_h, n_w))}
        maps = {}

        # 2) encoder per scale
        for p in self.patch_sizes:
            tokens, (n_h, n_w) = per_scale[p]
            for blk in self.encoders[str(p)]:
                tokens = blk(tokens, (n_h, n_w))
            # to map: (B,N,C)->(B,C,Hs,Ws)
            fmap = tokens_to_map(tokens, n_h, n_w)
            maps[p] = fmap  # (B, C, Hs, Ws)

        # 3) fuse with FPN (returns finest resolution map in 128 channels)
        fused = self.fpn(maps)  # (B, 128, H_min, W_min) where H_min=img/patch_min

        # 4) upsample to image size + gated skip + final logits
        # bring fused to full image size before head (head also upsamples to output_size)
        fused_up = F.interpolate(fused, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        out = self.head(fused_up, skip)
        return out

def build_model():
    model = VisionTransformerForSegmentationMultiScale(use_gradient_checkpointing=True)
    #model = torch.compile(model, dynamic=True)

    if getattr(settings, "load_from", None) is not None:
        model.load_state_dict(settings.load_from)

    return model
