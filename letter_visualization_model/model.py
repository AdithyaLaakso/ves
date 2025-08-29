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

def compute_edge_map(x):
    gray = x.mean(dim=1, keepdim=True)
    sobel_x = F.conv2d(gray, torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],
                       device=x.device, dtype=x.dtype), padding=1)
    sobel_y = F.conv2d(gray, torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]],
                       device=x.device, dtype=x.dtype), padding=1)
    edges = torch.sqrt(sobel_x**2 + sobel_y**2)
    edges = F.avg_pool2d(edges, 4, stride=4)  # downsample
    return edges

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=8, stride=8, embed_dim=128, in_chans=1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride)

    def forward(self, x):
        patches = self.proj(x)  # [B, C, H', W']
        B, C, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # [B, HW, C]
        return patches, (H, W)

class MultiScalePatchEmbed(nn.Module):
    def __init__(self, embed_dim=128, in_chans=1):
        super().__init__()
        csize, fsize = settings.patch_sizes
        self.coarse = PatchEmbed(csize, csize, embed_dim, in_chans)
        self.fine   = PatchEmbed(fsize, fsize, embed_dim, in_chans)
        self.type_embed = nn.Embedding(2, embed_dim)  # 0=coarse, 1=fine

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Coarse ---
        coarse_tokens, (Hc, Wc) = self.coarse(x)
        coarse_type = self.type_embed(torch.zeros(
            (B, coarse_tokens.size(1)), device=x.device, dtype=torch.long))
        coarse_tokens = coarse_tokens + coarse_type

        # --- Fine ---
        edges = compute_edge_map(x)   # [B,1,H/4,W/4]
        mask = edges > edges.mean(dim=[2,3], keepdim=True)
        fine_tokens, (Hf, Wf) = self.fine(x)

        mask_up = F.interpolate(mask.float(), size=(Hf, Wf), mode="nearest")
        mask_flat = mask_up.flatten(1).bool()

        fine_list = []
        for b in range(B):
            fine_list.append(fine_tokens[b][mask_flat[b]])
        fine_tokens = nn.utils.rnn.pad_sequence(fine_list, batch_first=True)
        fine_type = self.type_embed(torch.ones(
            (B, fine_tokens.size(1)), device=x.device, dtype=torch.long))
        fine_tokens = fine_tokens + fine_type

        tokens = torch.cat([coarse_tokens, fine_tokens], dim=1)  # [B, N_total, C]
        return tokens, (Hc, Wc), (Hf, Wf), mask_flat

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
            self.global_bias = nn.Parameter(torch.zeros(size, size, device=settings.device, dtype=dtype))
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
            rpb = self.relative_position_bias().to(dtype=attn_scores.dtype, device=settings.device)
            attn_scores = attn_scores + rpb.unsqueeze(0)

        gb = self._ensure_bias(B, self.num_heads, N, settings.device, attn_scores.dtype)
        attn_scores = attn_scores + gb.unsqueeze(0)

        attn = attn_scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

# Fix 1: Improved positional encoding with boundary normalization
def get_2d_sinusoidal_encoding_fixed(n_h, n_w, d_model):
    """Fixed version with better boundary handling"""
    assert d_model % 4 == 0

    pe = torch.zeros(n_h, n_w, d_model)
    d_model_quarter = d_model // 4

    # Use normalized positions to avoid boundary effects
    pos_row = torch.linspace(0, 1, n_h).unsqueeze(1)
    pos_col = torch.linspace(0, 1, n_w).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model_quarter, 2) *
                        -(math.log(10000.0) / d_model_quarter))

    pe[:, :, 0:d_model_quarter:2] = torch.sin(pos_row * div_term * math.pi).unsqueeze(1)
    pe[:, :, 1:d_model_quarter:2] = torch.cos(pos_row * div_term * math.pi).unsqueeze(1)
    pe[:, :, d_model_quarter:d_model_half:2] = torch.sin(pos_col * div_term * math.pi).unsqueeze(0)
    pe[:, :, d_model_quarter+1:d_model_half:2] = torch.cos(pos_col * div_term * math.pi).unsqueeze(0)

    # Normalize to reduce boundary effects
    pe = F.normalize(pe, dim=-1, p=2)

    return pe.view(-1, d_model)

class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_size,
                              kernel_size=patch_size, stride=patch_size)
        self.n_h = image_size // patch_size
        self.n_w = image_size // patch_size
        pe: torch.Tensor = get_2d_sinusoidal_encoding(self.n_h, self.n_w, embed_size)
        self.register_buffer("positional_encoding", pe, persistent=False)

    @torch.compile
    def forward(self, x: torch.Tensor):
        x = self.proj(x)                       # (B, embed, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)       # (B, N, embed)
        pos_enc = self.positional_encoding.unsqueeze(0)
        x = x + pos_enc

        # if settings.mode == settings.CLASSIFICATION or settings.mode == settings.MULTITASK:
        #     classification = self.classifier(x)
        #     return (x, (self.n_h, self.n_w)), classification
        # else:
        return x, (self.n_h, self.n_w)

class ConvDecoder(nn.Module):
    """Simple CNN for spatial reconstruction"""
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_chans, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

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
        x = checkpoint(lambda inp: self._mlp_forward(inp, grid_size), x, use_reentrant=True)
        x = checkpoint(self._attn_forward, x, use_reentrant=True)

        # x = self._attn_forward(x)
        # x = self._mlp_forward(x, grid_size)
        return x

class MultiScaleDecoder(nn.Module):
    def __init__(self, embed_dim=32, out_chans=1):
        super().__init__()
        self.out_chans = out_chans

        # Project coarse tokens -> coarse feature map
        self.coarse_proj = nn.Linear(embed_dim, out_chans)

        # Project fine tokens -> fine feature map
        self.fine_proj = nn.Linear(embed_dim, out_chans)

        # Convolutional fusion
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_chans * 2, out_chans * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans * 2, out_chans, 3, padding=1)
        )

    def forward(self, tokens, HcWc, HfWf, mask_flat, B):
        Hc, Wc = HcWc
        Hf, Wf = HfWf
        Nc = Hc * Wc

        coarse_tokens, fine_tokens = tokens[:, :Nc], tokens[:, Nc:]

        # --- Coarse branch ---
        coarse = self.coarse_proj(coarse_tokens)          # (B, Nc, C)
        coarse = coarse.view(B, Hc, Wc, self.out_chans)  # (B, Hc, Wc, C)
        coarse = coarse.permute(0, 3, 1, 2)              # (B, C, Hc, Wc)
        coarse_up = F.interpolate(coarse, size=(32, 32), mode='bilinear', align_corners=True)

        # --- Fine branch ---
        fine_map = torch.zeros(B, self.out_chans, Hf, Wf, device=tokens.device, dtype=tokens.dtype)

        for b in range(B):
            coords = mask_flat[b].nonzero(as_tuple=False).squeeze(-1)
            if coords.numel() == 0:
                continue
            cur_tokens = fine_tokens[b, :coords.numel()]
            fine_feats = self.fine_proj(cur_tokens)  # (num_tokens, C)
            for k, pos in enumerate(coords):
                row, col = divmod(pos.item(), Wf)
                fine_map[b, :, row, col] = fine_feats[k]

        fine_up = F.interpolate(fine_map, size=(32, 32), mode='bilinear', align_corners=True)

        # --- Fuse coarse + fine ---
        fused = torch.cat([coarse_up, fine_up], dim=1)  # (B, 2C, 32, 32)
        out = self.fuse_conv(fused)                     # (B, C, 32, 32)

        return out

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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: decoder feature (from projection path)
        # x: skip feature (original input / encoder feature)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SimpleAttention(nn.Module):
    """Simple attention without positional bias for irregular token sequences"""
    def __init__(self, dim, num_heads: int=1, qkv_bias: bool=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim[0] // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim[0], dim[0] * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim[0], dim[0])

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class SimpleMLP(nn.Module):
    """Simple MLP for irregular token sequences"""
    def __init__(self, embed_size, dropout=0.0, hidden_ratio=2):
        super().__init__()
        hidden = embed_size * hidden_ratio
        self.layers = nn.Sequential(
            nn.Linear(embed_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)

class SimpleEncoderBlock(nn.Module):
    """Simple encoder block for irregular token sequences"""
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.attn = SimpleAttention(dim=(embed_size, embed_size), num_heads=num_heads)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = SimpleMLP(embed_size, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class VisionTransformerForSegmentationMultiScale(nn.Module):
    def __init__(self, use_gradient_checkpointing=settings.use_gradient, num_classes=25):
        super().__init__()
        # read settings with sensible fallbacks
        self.output_size = settings.output_size
        self.in_channels = settings.in_channels
        self.out_channels = settings.out_channels
        self.image_size = settings.image_size
        self.embed_size = settings.embed_size
        self.num_blocks = settings.num_blocks
        self.num_heads = settings.num_heads
        self.dropout = settings.dropout
        self.patch_sizes = settings.patch_sizes
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Multiscale patch embedding
        self.encoder = MultiScalePatchEmbed(embed_dim=self.embed_size, in_chans=self.in_channels)

        # Coarse transformer with positional bias (for regular grid)
        coarse_patch_size = self.patch_sizes[0]
        n_h = self.image_size // coarse_patch_size
        n_w = self.image_size // coarse_patch_size
        window_size = (n_h, n_w)

        self.coarse_transformer = nn.ModuleList([
            SelfAttentionEncoderBlock(self.embed_size, self.num_heads, self.dropout,
                                      window_size=window_size,
                                      use_checkpoint=settings.use_gradient,
                                      mlp_dilation=2 if i % 2 == 0 else 3)
            for i in range(self.num_blocks)
        ])

        # Fine transformer without positional bias (for irregular tokens)
        self.fine_transformer = nn.ModuleList([
            SimpleEncoderBlock(self.embed_size, self.num_heads, self.dropout)
            for _ in range(self.num_blocks)
        ])

        # Multiscale decoder for segmentation
        self.decoder = MultiScaleDecoder(embed_dim=self.embed_size, out_chans=self.out_channels)

        # Classification head (hybrid: tokens + segmentation map)
        if settings.mode == settings.MULTITASK:
            self.classifier = HybridClassifier(embed_dim=self.embed_size, num_classes=num_classes)

    def forward(self, x):
        B = x.size(0)
        tokens, HcWc, HfWf, mask_flat = self.encoder(x)

        # Get dimensions
        Hc, Wc = HcWc
        Nc = Hc * Wc

        # Split tokens back to coarse and fine
        coarse_tokens = tokens[:, :Nc]
        fine_tokens = tokens[:, Nc:]

        # Process coarse tokens (regular grid)
        for blk in self.coarse_transformer:
            coarse_tokens = blk(coarse_tokens, (Hc, Wc))

        # Process fine tokens (irregular sequence)
        if fine_tokens.size(1) > 0:
            for blk in self.fine_transformer:
                fine_tokens = blk(fine_tokens)

        # Recombine the processed tokens
        processed_tokens = torch.cat([coarse_tokens, fine_tokens], dim=1)

        # Decode segmentation map
        out = self.decoder(processed_tokens, HcWc, HfWf, mask_flat, B)
        out = F.interpolate(out, size=(32, 32), mode="bilinear")

        if settings.mode == settings.MULTITASK:
            label = self.classifier(processed_tokens, out)  # <-- pass both
            return out, label
        elif settings.mode == settings.RECONSTRUCTION:
            return out
        else:
            raise ValueError(f"Unknown mode: {settings.mode}")

class HybridClassifier(nn.Module):
    """
    Combines global token features + segmentation map features for classification
    """
    def __init__(self, embed_dim=128, num_classes=25):
        super().__init__()
        # Token branch
        self.token_fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
        )

        # Segmentation map branch
        self.seg_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (16, 16, 16)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # -> (32, 4, 4)
        )

        # Fusion + output
        self.fc = nn.Sequential(
            nn.Linear(256 + 32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, tokens, seg_map):
        # Token branch (mean-pool over tokens)
        pooled = tokens.mean(dim=1)           # (B, embed_dim)
        token_feat = self.token_fc(pooled)    # (B, 256)

        # Segmentation branch
        seg_feat = self.seg_conv(seg_map)     # (B, 32, 4, 4)
        seg_feat = seg_feat.flatten(1)        # (B, 512)

        # Fuse
        fused = torch.cat([token_feat, seg_feat], dim=1)
        return self.fc(fused)

def build_model(compile_model=False, load_from=None, device=settings.device):
    model = VisionTransformerForSegmentationMultiScale(use_gradient_checkpointing=settings.use_gradient)

    if settings.load_from is not None:
        state_dict = torch.load(settings.load_from)
        model.load_state_dict(state_dict, strict=False)
    elif load_from is not None:
        state_dict = torch.load(load_from)
        model.load_state_dict(state_dict, strict=False)
    if compile_model:
        model = torch.compile(model, dynamic=True)

    return model.to(device)
