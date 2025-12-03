
from models.unet_vae import UNetVAE
from torchinfo import summary

# Create model
model = UNetVAE(
    input_channels=3,
    latent_dim=512,
    hidden_dims=[64, 128, 256, 512, 512],
    image_size=256,
    use_attention=True,
    use_skip_connections=True,
    pretrained_encoder=None
)

# Generate detailed summary
model_stats = summary(
    model,
    input_size=[(1, 3, 256, 256), (1, 1, 256, 256)],  # (batch, channels, height, width)
    col_names=['input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'],
    col_width=20,
    row_settings=['var_names'],
    verbose=2
)

print(model_stats)


from torchview import draw_graph
import torch
from models.unet_vae import UNetVAE

# Create model
model = UNetVAE(
    input_channels=3,
    latent_dim=512,
    hidden_dims=[64, 128, 256, 512, 512],
    image_size=256,
    use_attention=True,
    use_skip_connections=True,
    pretrained_encoder=None  # Non-pretrained
)

# Create dummy input
batch_size = 1
image = torch.randn(batch_size, 3, 256, 256)
mask = torch.randn(batch_size, 1, 256, 256)

# Generate visualization
model_graph = draw_graph(
    model, 
    input_data=[image, mask],
    expand_nested=True,
    graph_name='VAE_Model',
    save_graph=True,  # Saves as .png
    filename='vae_architecture',
    directory='./model_diagrams/'
)

# Display the graph (in Jupyter/Colab)
model_graph.visual_graph

# Render to file without displaying
model_graph.visual_graph.render(
    filename='model_diagrams/vae_architecture',
    format='png',
    cleanup=True,  # Remove the intermediate .gv file
    view=False  # Don't try to open the file
)
