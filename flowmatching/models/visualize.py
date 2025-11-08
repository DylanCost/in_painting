"""Visualization utility for U-Net architecture.

This script instantiates the U-Net model, performs a forward pass with dummy inputs,
and prints the shape of tensors at each layer/block to help understand the architecture.
"""
import torch
from src.models.unet import create_unet
import torchviz
import torchview

def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)

def print_section(title):
    """Print a section header."""
    print_separator()
    print(f" {title}")
    print_separator()

def visualize_model_architecture():
    """Visualize the U-Net model architecture and tensor shapes."""
    
    print_section("U-Net Model Architecture Visualization")
    
    # Model configuration
    batch_size = 2
    in_channels = 4  # RGB + mask
    out_channels = 3  # RGB velocity
    image_size = 128
    time_embed_dim = 256
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input channels: {in_channels} (RGB + mask)")
    print(f"  Output channels: {out_channels} (RGB velocity)")
    print(f"  Image size: {image_size}×{image_size}")
    print(f"  Time embedding dimension: {time_embed_dim}")
    
    # Create model
    print("\n" + "=" * 80)
    print("Creating U-Net model...")
    model = create_unet(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=64,
        time_embed_dim=time_embed_dim
    )
    
    # Model statistics
    num_params = model.get_num_parameters()
    model_size = model.get_model_size_mb()
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Model size: {model_size:.2f} MB")
    
    # Create dummy inputs
    print("\n" + "=" * 80)
    print("Creating dummy inputs...")
    
    # Input: [B, 4, H, W] - RGB image + binary mask
    x = torch.randn(batch_size, in_channels, image_size, image_size)
    print(f"  Input tensor shape: {list(x.shape)}")
    
    # Time: [B] - time values in [0, 1]
    t = torch.rand(batch_size)
    print(f"  Time tensor shape: {list(t.shape)}")
    
    # Forward pass with shape tracking
    print_section("Forward Pass - Tensor Shapes")
    
    model.eval()
    with torch.no_grad():
        # Time embedding
        print("\n1. Time Embedding:")
        time_emb = model.time_embedding(t)
        print(f"   Input time: {list(t.shape)}")
        print(f"   Time embedding: {list(time_emb.shape)}")
        
        # Initial convolution
        print("\n2. Initial Convolution:")
        print(f"   Input: {list(x.shape)}")
        x_init = model.init_conv(x, time_emb)
        print(f"   Output: {list(x_init.shape)}")
        
        # Encoder path
        print("\n3. Encoder Path (Downsampling):")
        x_enc = x_init
        skip_connections = [x_enc]
        
        for i, encoder_block in enumerate(model.encoder_blocks):
            x_enc = encoder_block(x_enc, time_emb)
            skip_connections.append(x_enc)
            print(f"   Level {i+1}: {list(x_enc.shape)}")
        
        skip_connections = skip_connections[:-1]
        
        # Bottleneck
        print("\n4. Bottleneck (Convolutions):")
        x_bottleneck = x_enc
        print(f"   Input: {list(x_bottleneck.shape)}")
        
        for j, block in enumerate(model.bottleneck):
            x_bottleneck = block(x_bottleneck, time_emb)
            print(f"   After DoubleConv {j + 1}: {list(x_bottleneck.shape)}")
        
        # Decoder path
        print("\n5. Decoder Path (Upsampling):")
        x_dec = x_bottleneck
        
        for i, decoder_block in enumerate(model.decoder_blocks):
            skip = skip_connections.pop()
            print(f"   Level {i+1}:")
            print(f"     Before upsampling: {list(x_dec.shape)}")
            print(f"     Skip connection: {list(skip.shape)}")
            x_dec = decoder_block(x_dec, skip, time_emb)
            print(f"     After upsampling: {list(x_dec.shape)}")
        
        # Output convolution
        print("\n6. Output Convolution:")
        print(f"   Input: {list(x_dec.shape)}")
        output = model.out_conv(x_dec)
        print(f"   Output: {list(output.shape)}")
        
        # Full forward pass
        print("\n" + "=" * 80)
        print("Full Forward Pass:")
        print(f"  Input: {list(x.shape)}")
        print(f"  Time: {list(t.shape)}")
        
        output_full = model(x, t)
        print(f"  Output: {list(output_full.shape)}")
        
        # Verify output shape
        expected_shape = [batch_size, out_channels, image_size, image_size]
        assert list(output_full.shape) == expected_shape, \
            f"Output shape mismatch! Expected {expected_shape}, got {list(output_full.shape)}"
        
        print(f"\n✓ Output shape verified: {list(output_full.shape)}")
        
        # Create a graphviz representation of the model
        model = create_unet(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=64,
            time_embed_dim=time_embed_dim
        )
        # graph = torchviz.make_dot(model(x, t), params=dict(model.named_parameters()))
        # graph.render("model_architecture", format="png")
        torchview.draw_graph(model, save_graph=True, input_data=(x, t), filename="model_architecture")
    
    # Architecture summary
    print_section("Architecture Summary")
    
    print("\nEncoder Levels:")
    channels = [64, 128, 256, 512]
    resolutions = [128, 64, 32, 16]
    for i, (ch, res) in enumerate(zip(channels, resolutions)):
        print(f"  Level {i+1}: {ch} channels, {res}×{res} resolution")
    
    print("\nBottleneck:")
    print(f"  512 channels, 8×8 resolution (3 convolution blocks)")
    
    print("\nDecoder Levels:")
    for i, (ch, res) in enumerate(zip(reversed(channels), [16, 32, 64, 128])):
        print(f"  Level {i+1}: {ch} channels, {res}×{res} resolution")
    
    print("\nKey Features:")
    print("  ✓ Time conditioning via sinusoidal embeddings")
    print("  ✓ Skip connections between encoder and decoder")
    print("  ✓ Pure convolution-based architecture")
    print("  ✓ GroupNorm for normalization")
    print("  ✓ SiLU activation functions")
    
    print_separator()
    print("Visualization complete!")
    print_separator()

if __name__ == "__main__":
    visualize_model_architecture()