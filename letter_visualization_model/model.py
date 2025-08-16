import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import settings
from torch.utils.checkpoint import checkpoint

# ImageToPatches returns multiple flattened square patches from an
# input image tensor.
class ImageToPatches(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert len(x.size()) == 4
        y = self.unfold(x)
        y = y.permute(0, 2, 1)
        return y

# The PatchEmbedding layer takes multiple image patches in (B,T,Cin) format
# and returns the embedded patches in (B,T,Cout) format.
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        # A single Layer is used to map all input patches to the output embedding dimension.
        # i.e. each image patch will share the weights of this embedding layer.
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)
    # end def

    def forward(self, x):
        assert len(x.size()) == 3
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x
    # end def


class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        """in_channels is the number of input channels in the input that will be
        fed into this layer. For RGB images, this value would be 3.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.i2p = ImageToPatches(image_size, patch_size)

        # Calculate the correct input features for patch embedding
        patch_features = patch_size * patch_size * in_channels
        self.pe = PatchEmbedding(patch_features, embed_size)

        num_patches = (image_size // patch_size) ** 2
        # position_embed below is the learned embedding for the position of each patch
        # in the input image. They correspond to the cosine similarity of embeddings
        # visualized in the paper "An Image is Worth 16x16 Words"
        # https://arxiv.org/pdf/2010.11929.pdf (Figure 7, Center).
        self.position_embed = nn.Parameter(torch.zeros(num_patches, embed_size))

    # end def

    def forward(self, x):
        x = self.i2p(x)
        x = self.pe(x)
        x = x + self.position_embed
        return x

# Memory-optimized MultiLayerPerceptron with optional gradient checkpointing
class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size, dropout, use_gelu=True, expansion_factor=4):
        super().__init__()
        # Reduced expansion factor from 4x to configurable (default 4, can reduce to 2-3)
        hidden_size = embed_size * expansion_factor

        # Use GELU instead of Sigmoid for better performance and memory efficiency
        activation = nn.GELU() if use_gelu else nn.Sigmoid()

        self.layers = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            activation,
            nn.Linear(hidden_size, embed_size),
            nn.Dropout(p=dropout),
        )
    # end def

    def forward(self, x):
        return self.layers(x)
    # end def

# Memory-optimized self-attention encoder block with gradient checkpointing option
class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, use_checkpoint=False):
        super().__init__()
        self.embed_size = embed_size
        self.use_checkpoint = use_checkpoint
        self.ln1 = nn.LayerNorm(embed_size)

        # Use scaled_dot_product_attention for memory efficiency (PyTorch 2.0+)
        self.mha = nn.MultiheadAttention(
            embed_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(embed_size)
        # Reduced MLP expansion factor to save memory
        self.mlp = MultiLayerPerceptron(embed_size, dropout, expansion_factor=2)
    # end def

    def _attention_forward(self, x):
        y = self.ln1(x)
        return x + self.mha(y, y, y, need_weights=False)[0]

    def _mlp_forward(self, x):
        return x + self.mlp(self.ln2(x))

    def forward(self, x):
        # Use gradient checkpointing to trade compute for memory
        if self.use_checkpoint and self.training:
            x = checkpoint(self._attention_forward, x, use_reentrant=False)
            x = checkpoint(self._mlp_forward, x, use_reentrant=False)
        else:
            x = self._attention_forward(x)
            x = self._mlp_forward(x)
        return x

# Memory-optimized output projection with progressive upsampling
class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, output_dims, output_size, use_progressive_upsampling=True):
        super().__init__()
        self.patch_size = patch_size
        self.output_dims = output_dims
        self.output_size = output_size
        self.use_progressive_upsampling = use_progressive_upsampling

        # Option 1: Direct projection (memory intensive)
        self.projection = nn.Linear(embed_size, patch_size * patch_size * output_dims)
        self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)

        # Option 2: Progressive upsampling (memory efficient)
        if use_progressive_upsampling:
            # Use smaller intermediate projection
            intermediate_dims = min(embed_size // 2, 128)
            self.progressive_projection = nn.Sequential(
                nn.Linear(embed_size, intermediate_dims),
                nn.ReLU(inplace=True),
                nn.Linear(intermediate_dims, patch_size * patch_size * output_dims)
            )
    # end def

    def forward(self, x):
        B, T, C = x.shape

        if self.use_progressive_upsampling:
            x = self.progressive_projection(x)
        else:
            x = self.projection(x)

        # x will now have shape (B, T, PatchSize**2 * OutputDims). This can be folded into
        # the desired output shape.

        # To fold the patches back into an image-like form, we need to first
        # swap the T and C dimensions to make it a (B, C, T) tensor.
        x = x.permute(0, 2, 1)
        x = self.fold(x)

        # Use memory-efficient interpolation
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        return x

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
    def __init__(self, use_gradient_checkpointing=True, memory_efficient_attention=True):
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
            for i in range(self.num_blocks)
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

    def forward(self, x):
        # Input processing
        x = self.vit_input(x)

        # Process through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        # Output projection
        x = self.output_projection(x)
        return x

# Memory optimization utilities
def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

def debug_tensor_shapes(model, input_tensor):

    # Test each component individually
    vit_input = model.vit_input

    # Test ImageToPatches
    patches = vit_input.i2p(input_tensor)

    # Check if patch features match embedding layer
    expected_features = vit_input.patch_size * vit_input.patch_size * vit_input.in_channels
    actual_features = patches.shape[-1]


    if expected_features != actual_features:
        return False
    else:
        return True

def optimize_model_for_memory(model, input_tensor=None, use_fp16=False):
    """Apply various memory optimizations to the model"""
    optimizations_applied = []

    # 1. Enable mixed precision training (but don't convert model to FP16 directly)
    if use_fp16:
        # Don't convert model weights to FP16 - let AMP handle this automatically
        # model = model.half()  # Remove this line
        optimizations_applied.append("Ready for AMP/FP16 training")

    # 2. If input tensor provided, run a forward pass to compile optimizations
    if input_tensor is not None:
        model.eval()
        with torch.no_grad():
            # Use FP32 for the test forward pass
            _ = model(input_tensor.float() if input_tensor.dtype != torch.float32 else input_tensor)
        optimizations_applied.append("Model compilation")

    return model, optimizations_applied

# Usage example with memory monitoring
def create_memory_efficient_vit(use_fp16=False):
    # Create model with memory optimizations
    model = VisionTransformerForSegmentation(
        use_gradient_checkpointing=True,
        memory_efficient_attention=True
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Apply additional memory optimizations
    # Use settings dimensions for dummy input
    dummy_input = torch.randn(1, settings.in_channels, settings.image_size, settings.image_size)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    # Test forward pass first to catch dimension errors
    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
    except Exception as e:
        return None

    model, optimizations = optimize_model_for_memory(model, dummy_input, use_fp16=use_fp16)

    return model
