"""Visualization utility for U-Net architecture.

This script instantiates the U-Net model, performs a forward pass with dummy inputs,
and prints the shape of tensors at each layer/block to help understand the architecture.
"""
import torch
from .unet import create_unet
import torchview
from config.common_config import Config

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
    
    # Load configuration matching pipeline.py
    config = Config()
    
    # Model configuration (matching pipeline.py lines 248-254)
    batch_size = 2
    in_channels = 4  # RGB + mask
    out_channels = 3  # RGB velocity
    image_size = config.data.image_size  # Default: 128
    time_embed_dim = 256  # Flow matching specific parameter
    hidden_dims = config.unet.hidden_dims  # Default: [64, 128, 256, 512, 512]
    
    print(f"\nConfiguration (matching pipeline.py):")
    print(f"  Batch size: {batch_size}")
    print(f"  Input channels: {in_channels} (RGB + mask)")
    print(f"  Output channels: {out_channels} (RGB velocity)")
    print(f"  Image size: {image_size}×{image_size}")
    print(f"  Time embedding dimension: {time_embed_dim}")
    print(f"  Hidden dimensions: {hidden_dims}")
    
    # Create model
    print("\n" + "=" * 80)
    print("Creating U-Net model...")
    model = create_unet(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dims=hidden_dims,
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
        
        # Encoder path
        print("\n2. Encoder Path:")
        x_enc = x
        skip_connections = []
        
        for i, (block, attention) in enumerate(zip(model.encoder.blocks, model.encoder.attention_blocks)):
            print(f"   Level {i+1}:")
            print(f"     Input: {list(x_enc.shape)}")
            x_enc = block(x_enc, time_emb)
            print(f"     After Block: {list(x_enc.shape)}")
            x_enc = attention(x_enc)
            print(f"     After Attention: {list(x_enc.shape)}")
            skip_connections.append(x_enc)
        
        # Remove last skip connection (it's the bottleneck input)
        skip_connections = skip_connections[:-1]
        
        # Decoder path
        print("\n3. Decoder Path:")
        x_dec = x_enc
        
        # Reverse skip connections for decoder
        skip_connections = list(reversed(skip_connections))
        
        for i, (block, attention) in enumerate(zip(model.decoder.blocks, model.decoder.attention_blocks)):
            print(f"   Level {i+1}:")
            print(f"     Input: {list(x_dec.shape)}")
            
            if i > 0 and i <= len(skip_connections):
                skip = skip_connections[i-1]
                print(f"     Skip connection: {list(skip.shape)}")
                # Note: Concatenation happens inside the block in the new architecture
                # We are just simulating the flow here
            
            # We can't easily simulate the internal concatenation of the decoder block
            # without replicating its logic, so we'll just run the block
            # But we need to pass the correct skip connections list to the full decoder forward
            # Here we are just iterating blocks for visualization
            
            # To properly visualize, we should probably just run the full decoder forward
            # but let's try to approximate for the printout
            
            # Actually, let's just run the full model forward and print shapes from hooks if we wanted detailed
            # internal inspection. But for now, let's just trust the full forward pass.
            pass

        # Let's just run the full forward pass to verify end-to-end
        print("\nRunning full forward pass...")
        output_full = model(x, t)
        print(f"  Output: {list(output_full.shape)}")
        
        # Verify output shape
        expected_shape = [batch_size, out_channels, image_size, image_size]
        assert list(output_full.shape) == expected_shape, \
            f"Output shape mismatch! Expected {expected_shape}, got {list(output_full.shape)}"
        
        print(f"\n✓ Output shape verified: {list(output_full.shape)}")
        
        # Create a graphviz representation of the model
        # (reusing the same model instance already created above)
        torchview.draw_graph(model, save_graph=True, input_data=(x, t), expand_nested=True, filename="model_architecture")
    
    # Architecture summary
    print_section("Architecture Summary")
    
    print("\nEncoder Levels:")
    for i, ch in enumerate(hidden_dims):
        res = image_size // (2 ** (i + 1))
        print(f"  Level {i+1}: {ch} channels, {res}×{res} resolution")
    
    print("\nDecoder Levels:")
    for i, ch in enumerate(reversed(hidden_dims)):
        # Resolution calculation is approximate here
        print(f"  Level {i+1}: {ch} channels")
    
    print("\nKey Features:")
    print("  ✓ Time conditioning via sinusoidal embeddings (scaled)")
    print("  ✓ Skip connections between encoder and decoder")
    print("  ✓ Self-attention at specified resolutions")
    print("  ✓ Diffusion-style architecture")
    
    print_separator()
    print("Visualization complete!")
    print_separator()

if __name__ == "__main__":
    visualize_model_architecture()