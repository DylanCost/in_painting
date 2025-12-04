# Triptych Generation Utility

The script [`flowmatching/generate_triptychs.py`](../flowmatching/generate_triptychs.py) creates
reproducible qualitative figures for paper-ready comparisons. It renders the first
N images from the CelebA **test** split (no shuffling), applies the manual masks
versioned in [`config/common_config.py`](../config/common_config.py), reconstructs each
sample with the latest trained checkpoint, and lays out two 4×3 canvases (4 columns,
3 rows) without any padding.

## Requirements
- CelebA dataset available under `./assets/datasets` (override via `--data-root`).
- A trained flow-matching checkpoint stored under `runs/flowmatching/*/checkpoints/`.
- Python environment with the project dependencies installed (activate `.venv` and run with `uv run` or `python`).

## Usage
```bash
uv run python -m flowmatching.generate_triptychs \
    --output-dir runs/flowmatching/triptychs \
    --num-examples 8 \
    --num-steps 100 \
    --progress
```

Key flags:
- `--checkpoint`: override the automatically discovered `best.ckpt`/`last.ckpt`.
- `--device`: force `cpu` or `cuda`; defaults to auto-detection.
- `--num-examples`: must be a multiple of 4. Defaults to 8 (two panels of four).
- `--download`: allow torchvision to download CelebA if missing.

## Outputs
- `triptych_panel_01.png` and `triptych_panel_02.png` (`512×384 px`), each column showing original → masked → reconstructed stacks.
- `triptych_metadata.json` capturing checkpoint path, mask version, dataset indices, filenames, and panel locations for audit/reproducibility.

## Reproducibility Notes
- Manual mask specifications live in [`config/common_config.py`](../config/common_config.py) (`ManualMaskSpec`, `TRIPTYCH_MASK_VERSION`).
- The script seeds NumPy/Torch, enforces deterministic sample ordering, and preserves observed pixels during sampling to keep iterations stable.
- Adjust or extend the mask registry before increasing `--num-examples` beyond the predefined eight entries.
