"""Streamlit app for interactive image inpainting visualization.

This app provides an interactive interface for testing the flow matching
inpainting model with adjustable parameters and real-time visualization.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

# Add project root to Python path to enable absolute imports
# This is necessary when running the script from within the flowmatching directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import torch
import torch.nn as nn
import time
import numpy as np
from PIL import Image
import io

# Import project modules
from flowmatching.models import create_unet
from flowmatching.data import CelebAInpainting
from flowmatching.flow import ODESampler, HeunSampler
from flowmatching.training.metrics import compute_psnr, compute_ssim, denormalize_image
from masking.mask_generator import MaskGenerator
import torchvision.transforms.functional as TF

try:
    from flowmatching.pipeline import PipelineConfig
except Exception:
    PipelineConfig = None


logger = logging.getLogger("flowmatching.app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Page configuration
st.set_page_config(
    page_title="Flow Matching Inpainting",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Visualization utility functions
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to PIL Image.

    Args:
        tensor: Image tensor in [C, H, W] format, values in [0, 1]

    Returns:
        PIL Image
    """
    return TF.to_pil_image(tensor.cpu())


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor.

    Args:
        pil_image: PIL Image

    Returns:
        Tensor in [C, H, W] format, values in [0, 1]
    """
    return TF.to_tensor(pil_image)


def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.4,
    color: tuple = (1.0, 0.0, 0.0),
) -> torch.Tensor:
    """Overlay a colored mask on an image.

    Args:
        image: Image tensor [B, C, H, W] in [0, 1]
        mask: Binary mask [B, 1, H, W]
        alpha: Transparency of the overlay
        color: RGB color tuple for the mask

    Returns:
        Image with mask overlay [B, C, H, W]
    """
    # Create colored mask
    colored_mask = torch.zeros_like(image)
    for i, c in enumerate(color):
        colored_mask[:, i : i + 1] = c * mask

    # Blend image and colored mask
    overlay = (1 - alpha) * image + alpha * colored_mask

    # Only apply overlay where mask is 1
    result = torch.where(mask.expand_as(image) > 0.5, overlay, image)

    return result


@dataclass
class CheckpointInfo:
    """Information about a discovered checkpoint."""
    path: Path
    run_timestamp: str
    checkpoint_type: str  # "best" or "last"
    display_name: str
    metrics: Optional[Dict[str, float]] = None


def load_run_config_snapshot(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load the config snapshot associated with a checkpoint if available."""
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / "config_snapshot.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read config snapshot at %s: %s", config_path, exc)
    return None


def infer_hidden_dims_from_state(state_dict: Dict[str, torch.Tensor]) -> List[int]:
    """Infer encoder hidden dimensions from a checkpoint state dict."""
    hidden_dims = []
    idx = 0
    while True:
        weight_key = f"encoder.blocks.{idx}.conv1.0.weight"
        if weight_key not in state_dict:
            break
        hidden_dims.append(state_dict[weight_key].shape[0])
        idx += 1
    return hidden_dims


def get_default_model_config() -> Dict[str, Any]:
    """Return baseline model hyperparameters shared with the training pipeline."""
    if PipelineConfig is not None:
        pipeline_config = PipelineConfig()
        hidden_dims = list(pipeline_config.common.unet.hidden_dims)
        image_size = pipeline_config.common.data.image_size
    else:
        hidden_dims = [64, 128, 256, 512, 512]
        image_size = 128

    return {
        "image_size": image_size,
        "in_channels": 4,
        "out_channels": 3,
        "hidden_dims": hidden_dims,
        "time_embed_dim": 256,
    }


def resolve_model_config(config_snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve the model configuration using pipeline defaults and run snapshots."""
    config = get_default_model_config()

    if config_snapshot:
        hidden_dims = (
            config_snapshot.get("common", {})
            .get("unet", {})
            .get("hidden_dims")
        )
        if hidden_dims:
            config["hidden_dims"] = hidden_dims

        image_size = (
            config_snapshot.get("common", {})
            .get("data", {})
            .get("image_size")
        )
        if image_size:
            config["image_size"] = image_size

    return config


def discover_checkpoints() -> List[CheckpointInfo]:
    """Discover all available checkpoints from training runs.
    
    Scans runs/flowmatching/{timestamp}/checkpoints/ for best.ckpt and last.ckpt
    files, optionally loading metrics from eval_results.json for display.
    
    Returns:
        List of CheckpointInfo objects, sorted by timestamp (newest first)
    """
    checkpoints = []
    runs_dir = Path("runs/flowmatching")
    
    if not runs_dir.exists():
        return checkpoints
    
    # Iterate through run directories (sorted newest first)
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
            
        timestamp = run_dir.name
        ckpt_dir = run_dir / "checkpoints"
        
        if not ckpt_dir.exists():
            continue
        
        # Try to load metrics for display
        metrics = None
        eval_results_path = run_dir / "eval_results.json"
        if eval_results_path.exists():
            try:
                with open(eval_results_path) as f:
                    results = json.load(f)
                    metrics = results.get("metrics_masked", {})
            except Exception:
                pass  # Metrics are optional
        
        # Check for best checkpoint
        best_ckpt = ckpt_dir / "best.ckpt"
        if best_ckpt.exists():
            display_name = f"{timestamp}/best"
            if metrics:
                psnr = metrics.get("psnr", 0)
                ssim = metrics.get("ssim", 0)
                mae = metrics.get("mae", None)
                if mae is not None:
                    display_name += f" (PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, MAE: {mae:.4f})"
                else:
                    display_name += f" (PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f})"
            
            checkpoints.append(CheckpointInfo(
                path=best_ckpt,
                run_timestamp=timestamp,
                checkpoint_type="best",
                display_name=display_name,
                metrics=metrics
            ))
        
        # Check for last checkpoint
        last_ckpt = ckpt_dir / "last.ckpt"
        if last_ckpt.exists():
            display_name = f"{timestamp}/last"
            if metrics:
                psnr = metrics.get("psnr", 0)
                ssim = metrics.get("ssim", 0)
                mae = metrics.get("mae", None)
                if mae is not None:
                    display_name += f" (PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, MAE: {mae:.4f})"
                else:
                    display_name += f" (PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f})"
            
            checkpoints.append(CheckpointInfo(
                path=last_ckpt,
                run_timestamp=timestamp,
                checkpoint_type="last",
                display_name=display_name,
                metrics=metrics
            ))
    
    return checkpoints


@st.cache_resource
def load_model(checkpoint_path: str = None, device: str = "cpu"):
    """Load the U-Net model with optional checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (None for untrained model)
        device: Device to load model on

    Returns:
        Tuple of (model, is_trained)
    """
    checkpoint_snapshot = None
    checkpoint_path_obj: Optional[Path] = None
    if checkpoint_path:
        checkpoint_path_obj = Path(checkpoint_path)
        checkpoint_snapshot = load_run_config_snapshot(checkpoint_path_obj)

    # Resolve model configuration using pipeline defaults (and checkpoint snapshot if available)
    model_config = resolve_model_config(checkpoint_snapshot)
    logger.info("Resolved model configuration: %s", model_config)

    # Create model
    model = create_unet(**model_config)

    is_trained = False
    diag_details = {
        "app_hidden_dims": list(model_config.get("hidden_dims", [])),
        "model_config": model_config,
    }
    if checkpoint_snapshot is not None:
        config_hidden_dims = (
            checkpoint_snapshot.get("common", {})
            .get("unet", {})
            .get("hidden_dims")
        )
        if config_hidden_dims:
            diag_details["config_hidden_dims"] = config_hidden_dims

    # Load checkpoint if provided
    if checkpoint_path_obj is not None and checkpoint_path_obj.exists():
        try:
            checkpoint = torch.load(
                checkpoint_path_obj, map_location=device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            inferred_hidden_dims = infer_hidden_dims_from_state(state_dict)
            if inferred_hidden_dims:
                diag_details["checkpoint_hidden_dims"] = inferred_hidden_dims

            logger.info(
                "Checkpoint diagnostics | app_hidden_dims=%s | checkpoint_hidden_dims=%s | config_hidden_dims=%s",
                diag_details.get("app_hidden_dims"),
                diag_details.get("checkpoint_hidden_dims"),
                diag_details.get("config_hidden_dims"),
            )

            model.load_state_dict(state_dict)
            is_trained = True
            st.success(f"✅ Loaded trained model from {checkpoint_path}")
        except Exception as e:
            logger.error("Failed to load checkpoint %s", checkpoint_path, exc_info=True)
            st.warning(f"⚠️ Could not load checkpoint: {e}")

            diag_lines = [
                f"App hidden_dims: {diag_details.get('app_hidden_dims')}",
                f"Checkpoint hidden_dims (inferred): {diag_details.get('checkpoint_hidden_dims', 'unknown')}",
                f"Config hidden_dims: {diag_details.get('config_hidden_dims', 'unknown')}",
            ]
            st.info("\n".join(diag_lines))
            st.info("Using untrained model with random weights")
    elif checkpoint_path:
        st.warning(f"⚠️ Checkpoint path not found: {checkpoint_path}")
        st.info("Using untrained model with random weights")
    else:
        st.info("ℹ️ Using untrained model with random weights for demonstration")

    model = model.to(device)
    model.eval()

    return model, is_trained


@st.cache_resource
def load_dataset(root: str = "./assets/datasets", split: str = "valid"):
    """Load the CelebA dataset.

    Args:
        root: Root directory for dataset
        split: Dataset split to use

    Returns:
        CelebAInpainting dataset
    """
    try:
        dataset = CelebAInpainting(
            root=root, split=split, image_size=128, download=True
        )
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def get_model_info(model: nn.Module) -> dict:
    """Get model information including parameter count.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "Total Parameters": f"{total_params:,}",
        "Trainable Parameters": f"{trainable_params:,}",
        "Model Size (MB)": f"{total_params * 4 / (1024**2):.2f}",
    }


def main():
    """Main Streamlit app."""

    # Title and description
    st.title("🎨 Flow Matching Image Inpainting")
    st.markdown(
        """
    Interactive visualization app for flow matching-based image inpainting.
    Adjust parameters in the sidebar and generate inpainting results in real-time.
    """
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"🖥️ Using device: **{device.upper()}**")

    # Model checkpoint selection
    st.sidebar.subheader("Model Settings")
    
    # Discover available checkpoints
    available_checkpoints = discover_checkpoints()
    
    if available_checkpoints:
        # Create display names for dropdown
        checkpoint_options = ["None (Untrained)"] + [
            ckpt.display_name for ckpt in available_checkpoints
        ]
        
        selected_name = st.sidebar.selectbox(
            "Model Checkpoint",
            checkpoint_options,
            help="Select a trained checkpoint or use untrained model",
        )
        
        # Map selection back to checkpoint path
        if selected_name == "None (Untrained)":
            checkpoint_path = None
        else:
            # Find the selected checkpoint
            selected_ckpt = next(
                ckpt for ckpt in available_checkpoints
                if ckpt.display_name == selected_name
            )
            checkpoint_path = str(selected_ckpt.path)
            
            # Display additional run info
            with st.sidebar.expander("📊 Run Details"):
                st.text(f"Timestamp: {selected_ckpt.run_timestamp}")
                st.text(f"Type: {selected_ckpt.checkpoint_type}")
                if selected_ckpt.metrics:
                    st.text(f"PSNR: {selected_ckpt.metrics.get('psnr', 'N/A'):.2f} dB")
                    st.text(f"SSIM: {selected_ckpt.metrics.get('ssim', 'N/A'):.4f}")
                    if 'mae' in selected_ckpt.metrics:
                        st.text(f"MAE: {selected_ckpt.metrics['mae']:.6f}")
                    if 'lpips' in selected_ckpt.metrics:
                        st.text(f"LPIPS: {selected_ckpt.metrics['lpips']:.4f}")
    else:
        st.sidebar.info("ℹ️ No trained checkpoints found in runs/flowmatching/")
        st.sidebar.caption("Train a model using: `python -m flowmatching.pipeline`")
        checkpoint_path = None

    # Load model
    model, is_trained = load_model(checkpoint_path, device)

    # Sampling settings
    st.sidebar.subheader("Sampling Settings")
    num_steps = st.sidebar.slider(
        "Number of Sampling Steps",
        min_value=10,
        max_value=1_000,
        value=100,
        step=10,
        help="More steps = better quality but slower",
    )

    sampler_type = st.sidebar.selectbox(
        "Sampler Type", ["Euler", "Heun"], help="Heun is more accurate but 2x slower"
    )

    # Mask settings
    st.sidebar.subheader("Mask Settings")
    min_mask_size = st.sidebar.slider(
        "Minimum Mask Size",
        min_value=16,
        max_value=96,
        value=16,
        step=8,
        help="Minimum dimension of rectangular mask",
    )

    max_mask_size = st.sidebar.slider(
        "Maximum Mask Size",
        min_value=min_mask_size,
        max_value=96,
        value=64,
        step=8,
        help="Maximum dimension of rectangular mask",
    )

    # Random seed
    use_seed = st.sidebar.checkbox("Use Random Seed", value=False)
    if use_seed:
        seed = st.sidebar.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            help="For reproducible results",
        )
    else:
        seed = None

    # Action buttons
    st.sidebar.subheader("Actions")
    col1, col2 = st.sidebar.columns(2)
    generate_sample = col1.button("🎲 New Sample", use_container_width=True)
    generate_mask = col2.button("🎭 New Mask", use_container_width=True)

    # Initialize session state
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_mask" not in st.session_state:
        st.session_state.current_mask = None
    if "mask_generator" not in st.session_state:
        st.session_state.mask_generator = MaskGenerator(
            mask_type="random",
            min_size=min_mask_size,
            max_size=max_mask_size,
            deterministic=False,
        )

    # Update mask generator if settings changed
    if (
        st.session_state.mask_generator.min_size != min_mask_size
        or st.session_state.mask_generator.max_size != max_mask_size
    ):
        st.session_state.mask_generator = MaskGenerator(
            mask_type="random",
            min_size=min_mask_size,
            max_size=max_mask_size,
            deterministic=False,
        )

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🖼️ Inpainting", "📊 Model Info", "ℹ️ About"])

    with tab1:
        # Image source selection
        st.subheader("Image Source")
        image_source = st.radio(
            "Select image source:",
            ["Random from Dataset", "Upload Image"],
            horizontal=True,
        )

        # Load or upload image
        if image_source == "Random from Dataset":
            dataset = load_dataset()

            if dataset is not None and (
                generate_sample or st.session_state.current_image is None
            ):
                # Set seed if requested
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                # Get random sample
                idx = np.random.randint(0, len(dataset))
                sample = dataset[idx]
                st.session_state.current_image = sample["image"]

                # Generate new mask
                mask = st.session_state.mask_generator.generate(shape=(1, 128, 128))
                # Squeeze to get [1, H, W]
                st.session_state.current_mask = mask.squeeze(0)

        else:  # Upload Image
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg"],
                help="Upload a face image for inpainting",
            )

            if uploaded_file is not None:
                # Load and process uploaded image
                pil_image = Image.open(uploaded_file).convert("RGB")
                pil_image = pil_image.resize((128, 128))

                # Convert to tensor and normalize
                image_tensor = pil_to_tensor(pil_image)
                # Normalize to [-1, 1]
                image_tensor = (image_tensor - 0.5) / 0.5

                st.session_state.current_image = image_tensor

                # Generate mask if needed
                if st.session_state.current_mask is None or generate_mask:
                    mask = st.session_state.mask_generator.generate(shape=(1, 128, 128))
                    # Squeeze to get [1, H, W]
                    st.session_state.current_mask = mask.squeeze(0)

        # Generate new mask if requested
        if generate_mask and st.session_state.current_image is not None:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)

            mask = st.session_state.mask_generator.generate(shape=(1, 128, 128))
            # Squeeze to get [1, H, W]
            st.session_state.current_mask = mask.squeeze(0)

        # Process and display results
        if st.session_state.current_image is not None:
            image = st.session_state.current_image
            mask = st.session_state.current_mask

            # Add batch dimension
            image_batch = image.unsqueeze(0).to(device)
            mask_batch = mask.unsqueeze(0).to(device)

            # Create masked image
            masked_input = (1 - mask_batch) * image_batch

            # Run inference
            st.subheader("🔄 Running Inference...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create sampler
            if sampler_type == "Euler":
                sampler = ODESampler(model, num_steps=num_steps, device=device)
            else:
                sampler = HeunSampler(model, num_steps=num_steps, device=device)

            # Measure inference time
            start_time = time.time()

            with torch.no_grad():
                inpainted = sampler.sample(masked_input, mask_batch)

            inference_time = time.time() - start_time

            progress_bar.progress(100)
            status_text.success(f"✅ Inference completed in {inference_time:.2f}s")

            # Denormalize images for display
            image_display = denormalize_image(image_batch)
            masked_display = denormalize_image(masked_input)
            inpainted_display = denormalize_image(inpainted)

            # Create mask overlay
            mask_overlay = overlay_mask(image_display, mask_batch, alpha=0.4)

            # Display results in columns
            st.subheader("📸 Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Original Image**")
                st.image(tensor_to_pil(image_display[0]), use_container_width=True)

            with col2:
                st.markdown("**Masked Image**")
                st.image(tensor_to_pil(mask_overlay[0]), use_container_width=True)

            with col3:
                st.markdown("**Inpainted Result**")
                st.image(tensor_to_pil(inpainted_display[0]), use_container_width=True)

            # Compute metrics
            st.subheader("📊 Metrics")

            # Compute metrics on masked region only
            psnr_masked = compute_psnr(
                inpainted_display, image_display, mask=mask_batch, max_val=1.0
            )

            ssim_masked = compute_ssim(
                inpainted_display, image_display, mask=mask_batch, data_range=1.0
            )

            # Compute metrics on full image
            psnr_full = compute_psnr(
                inpainted_display, image_display, mask=None, max_val=1.0
            )

            ssim_full = compute_ssim(
                inpainted_display, image_display, mask=None, data_range=1.0
            )

            # Calculate mask ratio
            mask_ratio = mask_batch.mean().item() * 100

            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("PSNR (Masked)", f"{psnr_masked:.2f} dB")
                st.caption(f"Full: {psnr_full:.2f} dB")

            with col2:
                st.metric("SSIM (Masked)", f"{ssim_masked:.4f}")
                st.caption(f"Full: {ssim_full:.4f}")

            with col3:
                st.metric("Mask Ratio", f"{mask_ratio:.1f}%")

            with col4:
                st.metric("Inference Time", f"{inference_time:.2f}s")

            # Save results option
            st.subheader("💾 Save Results")
            if st.button("Download Comparison Image"):
                # Create comparison grid
                comparison = torch.cat(
                    [image_display[0], mask_overlay[0], inpainted_display[0]], dim=2
                )  # Concatenate horizontally

                # Convert to PIL and save to buffer
                comparison_pil = tensor_to_pil(comparison)
                buf = io.BytesIO()
                comparison_pil.save(buf, format="PNG")
                buf.seek(0)

                st.download_button(
                    label="📥 Download PNG",
                    data=buf,
                    file_name="inpainting_comparison.png",
                    mime="image/png",
                )

        else:
            st.info(
                "👆 Select an image source and click 'New Sample' or upload an image to begin"
            )

    with tab2:
        st.subheader("🏗️ Model Architecture")

        # Model info
        model_info = get_model_info(model)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parameters", model_info["Total Parameters"])
        with col2:
            st.metric("Trainable Parameters", model_info["Trainable Parameters"])
        with col3:
            st.metric("Model Size", model_info["Model Size (MB)"])

        # Model status
        st.subheader("📈 Model Status")
        if is_trained:
            st.success("✅ Using trained model checkpoint")
        else:
            st.warning("⚠️ Using untrained model with random weights")
            st.info(
                """
            The model is using random weights for demonstration purposes.
            Results will not be meaningful until the model is trained.
            
            To train the model, run:
            ```bash
            uv run python train.py
            ```
            """
            )

        # Architecture details
        with st.expander("🔍 View Model Architecture"):
            st.code(str(model), language="python")

    with tab3:
        st.subheader("ℹ️ About This App")

        st.markdown(
            """
        ### Flow Matching Image Inpainting
        
        This application demonstrates **flow matching** for image inpainting,
        a generative modeling approach that learns to transform masked images
        into complete images through continuous normalizing flows.
        
        #### Key Features:
        - 🎨 **Interactive Inpainting**: Generate inpainting results in real-time
        - 🎛️ **Adjustable Parameters**: Control sampling steps and mask sizes
        - 📊 **Real-time Metrics**: View PSNR and SSIM metrics
        - 🖼️ **Multiple Sources**: Use dataset images or upload your own
        - 💾 **Save Results**: Download comparison images
        
        #### How It Works:
        1. **Masking**: A rectangular region is randomly masked in the image
        2. **Flow Matching**: The model learns a velocity field that transforms
           the masked image to the complete image
        3. **ODE Sampling**: At inference, we integrate the learned ODE to
           generate the inpainted result
        
        #### Model Architecture:
        - **U-Net** with attention mechanisms
        - **Time embeddings** for flow matching
        - **Conditional** on mask and observed pixels
        
        #### Metrics:
        - **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
        - **SSIM**: Structural Similarity Index (closer to 1 is better)
        
        #### Usage Tips:
        - Start with 100 sampling steps for a balance of speed and quality
        - Use Heun sampler for better quality (but slower)
        - Adjust mask size to control difficulty
        - Use random seed for reproducible results
        
        ---
        
        **Note**: This is a demonstration app. For best results, train the model
        on the CelebA dataset first using `train.py`.
        """
        )


if __name__ == "__main__":
    main()
