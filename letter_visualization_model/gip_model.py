import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import settings
from torch.utils.checkpoint import checkpoint

# Enhanced ImageToPatches with multiple patch sizes
class MultiScaleImageToPatches(nn.Module):
    def __init__(self, image_size, patch_sizes):
        super().__init__()
        self.image_size = image_size
        self.patch_sizes = patch_sizes
        self.unfolds = nn.ModuleDict({
            f'patch_{ps}': nn.Unfold(kernel_size=ps, stride=ps)
            for ps in patch_sizes
        })

    def forward(self, x):
        assert len(x.size()) == 4
        multi_scale_patches = {}

        for ps in self.patch_sizes:
            patches = self.unfolds[f'patch_{ps}'](x)
            patches = patches.permute(0, 2, 1)  # (B, num_patches, patch_features)
            multi_scale_patches[ps] = patches

        return multi_scale_patches

# 2D Sinusoidal Position Encoding
class SinusoidalPosition2D(nn.Module):
    def __init__(self, embed_size, temperature=10000):
        super().__init__()
        self.embed_size = embed_size
        self.temperature = temperature

    def forward(self, h, w):
        """Generate 2D sinusoidal position encoding for h x w grid"""
        device = next(self.parameters()).device if hasattr(self, '_parameters') else 'cpu'

        y_embed = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).repeat(h, 1)

        if self.embed_size % 4 != 0:
            raise ValueError("embed_size must be divisible by 4 for 2D sinusoidal encoding")

        dim_t = torch.arange(self.embed_size // 4, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * dim_t / (self.embed_size // 4))

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[..., ::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., ::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1).flatten(0, 1)  # (h*w, embed_size)
        return pos

# Learnable 2D Position Embedding
class Learnable2DPositionEmbedding(nn.Module):
    def __init__(self, embed_size, max_h, max_w):
        super().__init__()
        self.embed_size = embed_size
        self.row_embed = nn.Parameter(torch.randn(max_h, embed_size // 2))
        self.col_embed = nn.Parameter(torch.randn(max_w, embed_size // 2))

    def forward(self, h, w):
        """Generate learnable 2D position embedding for h x w grid"""
        row_pos = self.row_embed[:h].unsqueeze(1).repeat(1, w, 1)  # (h, w, embed_size//2)
        col_pos = self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1)  # (h, w, embed_size//2)
        pos = torch.cat([row_pos, col_pos], dim=-1).flatten(0, 1)  # (h*w, embed_size)
        return pos

# Relative Position Encoding
class RelativePositionEncoding(nn.Module):
    def __init__(self, embed_size, max_relative_position):
        super().__init__()
        self.embed_size = embed_size
        self.max_relative_position = max_relative_position

        # Relative position embeddings for both dimensions
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, seq_len):
        """Generate relative position encoding matrix"""
        device = self.relative_position_embeddings.weight.device
        range_vec = torch.arange(seq_len, device=device)
        distance_mat = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        distance_mat = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        distance_mat = distance_mat + self.max_relative_position

        return self.relative_position_embeddings(distance_mat)

# Enhanced PatchEmbedding with multi-scale support
class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, patch_sizes, in_channels, embed_size):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.in_channels = in_channels
        self.embed_size = embed_size

        # Separate embedding layers for each patch size
        self.embed_layers = nn.ModuleDict({
            f'embed_{ps}': nn.Linear(
                in_features=ps * ps * in_channels,
                out_features=embed_size
            )
            for ps in patch_sizes
        })

        # Scale embeddings to distinguish patch sizes
        self.scale_embeddings = nn.ParameterDict({
            f'scale_{ps}': nn.Parameter(torch.randn(1, 1, embed_size))
            for ps in patch_sizes
        })

    def forward(self, multi_scale_patches):
        embedded_patches = {}

        for ps, patches in multi_scale_patches.items():
            embedded = self.embed_layers[f'embed_{ps}'](patches)
            embedded = embedded + self.scale_embeddings[f'scale_{ps}']
            embedded_patches[ps] = embedded

        return embedded_patches

# Enhanced VisionTransformerInput with multi-scale support
class MultiScaleVisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_sizes, in_channels, embed_size, position_type='learnable'):
        super().__init__()
        self.image_size = image_size
        self.patch_sizes = patch_sizes
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.position_type = position_type

        self.i2p = MultiScaleImageToPatches(image_size, patch_sizes)
        self.pe = MultiScalePatchEmbedding(patch_sizes, in_channels, embed_size)

        # Position encodings for different patch sizes
        self.position_encodings = nn.ModuleDict()

        for ps in patch_sizes:
            num_patches_h = image_size // ps
            num_patches_w = image_size // ps

            if position_type == 'sinusoidal':
                self.position_encodings[f'pos_{ps}'] = SinusoidalPosition2D(embed_size)
            elif position_type == 'learnable':
                self.position_encodings[f'pos_{ps}'] = Learnable2DPositionEmbedding(
                    embed_size, num_patches_h, num_patches_w
                )
            else:  # Simple learned embeddings (original)
                num_patches = num_patches_h * num_patches_w
                self.position_encodings[f'pos_{ps}'] = nn.Parameter(
                    torch.randn(num_patches, embed_size)
                )

    def forward(self, x):
        multi_scale_patches = self.i2p(x)
        embedded_patches = self.pe(multi_scale_patches)

        # Add position encodings
        for ps in self.patch_sizes:
            num_patches_h = self.image_size // ps
            num_patches_w = self.image_size // ps

            if self.position_type in ['sinusoidal', 'learnable']:
                pos_encoding = self.position_encodings[f'pos_{ps}'](num_patches_h, num_patches_w)
                embedded_patches[ps] = embedded_patches[ps] + pos_encoding.to(embedded_patches[ps].device)
            else:
                embedded_patches[ps] = embedded_patches[ps] + self.position_encodings[f'pos_{ps}']

        return embedded_patches

# Dilated MLP for larger receptive fields
class DilatedMLP(nn.Module):
    def __init__(self, embed_size, dropout, expansion_factor=4, dilation_rates=[1, 2, 4]):
        super().__init__()
        hidden_size = embed_size * expansion_factor

        # Multiple dilated convolution branches
        self.dilated_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, hidden_size // len(dilation_rates)),
                nn.GELU(),
                nn.Linear(hidden_size // len(dilation_rates), embed_size)
            ) for _ in dilation_rates
        ])

        self.dropout = nn.Dropout(p=dropout)
        self.combine = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # For simplicity, treating dilated convolutions as parallel linear layers
        # In practice, you might reshape to spatial dimensions and use Conv1d with dilation
        branch_outputs = [branch(x) for branch in self.dilated_branches]

        # Combine branches
        combined = torch.stack(branch_outputs, dim=-1).mean(dim=-1)
        output = self.combine(combined)
        return self.dropout(output)

# Enhanced self-attention with relative position encoding
class EnhancedSelfAttentionBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, use_relative_pos=True, use_checkpoint=False):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.use_relative_pos = use_relative_pos
        self.use_checkpoint = use_checkpoint

        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = nn.MultiheadAttention(
            embed_size, num_heads, dropout=dropout, batch_first=True
        )

        if use_relative_pos:
            self.relative_pos_encoding = RelativePositionEncoding(embed_size, max_relative_position=32)

        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = DilatedMLP(embed_size, dropout)

    def _attention_forward(self, x):
        y = self.ln1(x)
        attn_output, _ = self.mha(y, y, y, need_weights=False)

        # Add relative position encoding to attention
        if self.use_relative_pos:
            seq_len = x.size(1)
            rel_pos = self.relative_pos_encoding(seq_len)
            # Simplified relative position integration
            attn_output = attn_output + rel_pos.mean(dim=1).unsqueeze(0).expand_as(attn_output)

        return x + attn_output

    def _mlp_forward(self, x):
        return x + self.mlp(self.ln2(x))

    def forward(self, x):
        if self.use_checkpoint and self.training:
            x = checkpoint(self._attention_forward, x, use_reentrant=False)
            x = checkpoint(self._mlp_forward, x, use_reentrant=False)
        else:
            x = self._attention_forward(x)
            x = self._mlp_forward(x)
        return x

# Feature Pyramid Network for multi-scale feature fusion
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, embed_size, patch_sizes):
        super().__init__()
        self.embed_size = embed_size
        self.patch_sizes = sorted(patch_sizes)

        # Lateral connections for feature fusion
        self.lateral_convs = nn.ModuleDict({
            f'lateral_{ps}': nn.Linear(embed_size, embed_size)
            for ps in patch_sizes
        })

        # Top-down pathway
        self.fpn_convs = nn.ModuleDict({
            f'fpn_{ps}': nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(inplace=True),
                nn.Linear(embed_size, embed_size)
            )
            for ps in patch_sizes
        })

    def forward(self, multi_scale_features):
        # Process features from coarse to fine
        laterals = {}
        for ps in self.patch_sizes:
            laterals[ps] = self.lateral_convs[f'lateral_{ps}'](multi_scale_features[ps])

        # Top-down fusion
        fpn_features = {}
        prev_feature = None

        for ps in reversed(self.patch_sizes):  # Start from coarsest
            if prev_feature is not None:
                # Upsample previous feature to match current scale
                # Simple averaging for demonstration (in practice, use proper upsampling)
                upsampled = prev_feature.repeat_interleave(4, dim=1)[:, :laterals[ps].size(1), :]
                laterals[ps] = laterals[ps] + upsampled

            fpn_features[ps] = self.fpn_convs[f'fpn_{ps}'](laterals[ps])
            prev_feature = fpn_features[ps]

        return fpn_features

# Decoder with skip connections
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EnhancedSelfAttentionBlock(embed_size, num_heads, dropout, use_relative_pos=True)
            for _ in range(num_layers)
        ])

        # Skip connection processors
        self.skip_processors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_size),
                nn.Linear(embed_size, embed_size),
                nn.GELU()
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_features=None):
        outputs = []

        for i, layer in enumerate(self.layers):
            # Process skip connections from encoder
            if encoder_features and i < len(encoder_features):
                skip = self.skip_processors[i](encoder_features[i])
                x = x + skip

            x = layer(x)
            outputs.append(x)

        return x, outputs

# Progressive upsampling decoder
class ProgressiveUpsamplingDecoder(nn.Module):
    def __init__(self, image_size, patch_sizes, embed_size, output_dims, output_size):
        super().__init__()
        self.image_size = image_size
        self.patch_sizes = sorted(patch_sizes, reverse=True)  # Start from coarsest
        self.embed_size = embed_size
        self.output_dims = output_dims
        self.output_size = output_size

        # Simple upsampling modules for each scale
        self.upsample_modules = nn.ModuleDict()
        self.intermediate_heads = nn.ModuleDict()

        for ps in self.patch_sizes:
            # Direct upsampling from patch embeddings to output
            self.upsample_modules[f'upsample_{ps}'] = nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(inplace=True),
                nn.Linear(embed_size, ps * ps * output_dims)
            )

            # Intermediate supervision head
            self.intermediate_heads[f'head_{ps}'] = nn.Sequential(
                nn.Conv2d(output_dims, output_dims, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_dims, output_dims, 1)
            )

        # Multi-scale fusion module
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(output_dims * len(patch_sizes), output_dims, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dims, output_dims, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dims, output_dims, 1)
        )

    def patches_to_image(self, patches, patch_size, image_size):
        """Convert patches back to image format"""
        B, num_patches, patch_features = patches.shape
        patches_per_dim = image_size // patch_size

        # Ensure we have the right number of patches
        expected_patches = patches_per_dim * patches_per_dim
        if num_patches != expected_patches:
            # Pad or truncate if necessary
            if num_patches < expected_patches:
                padding = torch.zeros(B, expected_patches - num_patches, patch_features,
                                    device=patches.device, dtype=patches.dtype)
                patches = torch.cat([patches, padding], dim=1)
            else:
                patches = patches[:, :expected_patches, :]

        # Reshape to image format
        patches = patches.view(B, patches_per_dim, patches_per_dim,
                              patch_size, patch_size, self.output_dims)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        image = patches.view(B, self.output_dims, image_size, image_size)

        return image

    def forward(self, fpn_features):
        scale_outputs = []
        intermediate_outputs = []

        # Process each scale independently
        for ps in self.patch_sizes:
            features = fpn_features[ps]  # (B, num_patches, embed_size)

            # Upsample to patch format
            upsampled_patches = self.upsample_modules[f'upsample_{ps}'](features)

            # Convert patches to image
            scale_image = self.patches_to_image(upsampled_patches, ps, self.image_size)

            # Apply intermediate head
            processed_image = self.intermediate_heads[f'head_{ps}'](scale_image)

            # Resize to target output size
            if processed_image.shape[-1] != self.output_size:
                processed_image = F.interpolate(
                    processed_image, size=(self.output_size, self.output_size),
                    mode='bilinear', align_corners=False
                )

            scale_outputs.append(processed_image)
            intermediate_outputs.append(processed_image)

        # Fuse multi-scale outputs
        if len(scale_outputs) > 1:
            # Concatenate all scales
            multi_scale_features = torch.cat(scale_outputs, dim=1)
            final_output = self.fusion_conv(multi_scale_features)
        else:
            final_output = scale_outputs[0]

        return final_output, intermediate_outputs

# Enhanced Vision Transformer for Segmentation
class EnhancedVisionTransformerForSegmentation(nn.Module):
    def __init__(self, use_gradient_checkpointing=True, position_type='learnable'):
        super().__init__()
        self.output_size = settings.output_size
        self.in_channels = settings.in_channels
        self.out_channels = settings.out_channels
        self.image_size = settings.image_size
        self.patch_sizes = getattr(settings, 'patch_sizes', [8, 16, 32])
        self.embed_size = settings.embed_size
        self.num_blocks = settings.num_blocks
        self.num_heads = settings.num_heads
        self.dropout = settings.dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Multi-scale input processing
        self.vit_input = MultiScaleVisionTransformerInput(
            self.image_size, self.patch_sizes, self.in_channels,
            self.embed_size, position_type=position_type
        )

        # Separate encoders for each scale
        self.encoders = nn.ModuleDict({
            f'encoder_{ps}': nn.ModuleList([
                EnhancedSelfAttentionBlock(
                    self.embed_size, self.num_heads, self.dropout,
                    use_relative_pos=True, use_checkpoint=use_gradient_checkpointing
                )
                for _ in range(self.num_blocks)
            ])
            for ps in self.patch_sizes
        })

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(self.embed_size, self.patch_sizes)

        # Decoder with skip connections
        self.decoder = TransformerDecoder(
            self.embed_size, num_layers=4, num_heads=self.num_heads, dropout=self.dropout
        )

        # Progressive upsampling output
        self.output_decoder = ProgressiveUpsamplingDecoder(
            self.image_size, self.patch_sizes, self.embed_size,
            self.out_channels, self.output_size
        )

    def forward(self, x, return_dict=False):
        # Multi-scale input processing
        embedded_patches = self.vit_input(x)

        # Encode each scale separately and collect skip connections
        encoded_features = {}
        all_encoder_outputs = {}

        for ps in self.patch_sizes:
            features = embedded_patches[ps]
            encoder_outputs = []

            for block in self.encoders[f'encoder_{ps}']:
                features = block(features)
                encoder_outputs.append(features)

            encoded_features[ps] = features
            all_encoder_outputs[ps] = encoder_outputs

        # Feature pyramid fusion
        fpn_features = self.fpn(encoded_features)

        # Use finest scale features for decoder (with skip connections from all scales)
        finest_ps = min(self.patch_sizes)
        decoder_input = fpn_features[finest_ps]

        # Collect skip connections from all scales (use finest scale encoder outputs)
        skip_connections = all_encoder_outputs[finest_ps]

        # Decode with skip connections
        decoded_features, decoder_outputs = self.decoder(decoder_input, skip_connections)

        # Add decoded features back to FPN features for final output
        fpn_features[finest_ps] = decoded_features

        # Progressive upsampling to final output
        final_output, intermediate_outputs = self.output_decoder(fpn_features)

        if return_dict:
            return {
                'final_output': final_output,
                'intermediate_outputs': intermediate_outputs,
                'fpn_features': fpn_features
            }
        else:
            # Return just the final output for standard training
            return final_output

def debug_tensor_shapes(model, input_tensor):
    """Debug function to check tensor shapes throughout the network"""
    try:
        with torch.no_grad():
            output = model(input_tensor)
            print(f"Final output shape: {output['final_output'].shape}")
            print(f"Number of intermediate outputs: {len(output['intermediate_outputs'])}")
            for i, intermediate in enumerate(output['intermediate_outputs']):
                print(f"Intermediate output {i} shape: {intermediate.shape}")
        return True
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return False

def create_enhanced_memory_efficient_vit(use_fp16=False, position_type='learnable'):
    """Create the enhanced multi-scale Vision Transformer"""
    model = EnhancedVisionTransformerForSegmentation(
        use_gradient_checkpointing=True,
        position_type=position_type
    )

    if hasattr(settings, 'load_from') and settings.load_from is not None:
        model.load_state_dict(settings.load_from)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    return model
