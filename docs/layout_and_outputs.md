# Project layout and per-model pipeline outputs

Goals
- Keep model implementations black-boxed; centralize only shared config.
- Provide per-model pipeline.py that trains and evaluates on CelebA.
- Standardize run outputs for easy cross-model comparison.

Directory layout overview

```
in_painting/
├─ common_config.py
├─ runs/
│  └─ {model}/
│     └─ {timestamp}/
│        ├─ eval_results.json
│        ├─ config_snapshot.json
│        ├─ training.log
│        ├─ learning_curves.png
│        ├─ history.csv (optional)
│        ├─ checkpoints/
│        │  ├─ best.ckpt
│        │  └─ last.ckpt
│        └─ examples/
│           ├─ triptych_0001.png
│           └─ examples_grid.png
├─ data/
│  ├─ __init__.py
│  └─ celeba_dataset.py
├─ evaluation/
│  ├─ metrics.py
│  └─ results.py (optional)
├─ scripts/
│  ├─ train.py
│  └─ ...
├─ flowmatching/
│  ├─ pipeline.py
│  └─ ...
├─ vae/
│  ├─ pipeline.py
│  └─ ...
└─ diffusion/
   ├─ pipeline.py
   └─ ...
```

Per-model pipeline contract (minimal)
- Self-contained script [pipeline.py](flowmatching/pipeline.py) per model; no shared engine and no exported Python API required.
- Responsibilities:
  - Load and optionally merge shared settings from [common_config.py](common_config.py).
  - Train the model; persist checkpoints, logs, and learning curves.
  - Evaluate on the test split; compute masked-region metrics; save example triptychs.
  - Write standardized results artifacts under [runs/{model}/{timestamp}/](runs/).

Run outputs and conventions
- Base directory: [runs/{model}/{timestamp}/](runs/)
- Required:
  - eval_results.json
  - config_snapshot.json
  - training.log
  - learning_curves.png
  - checkpoints/
  - examples/ (triptych_{idx}.png; examples_grid.png optional)
- Optional:
  - history.csv (per-epoch metrics/training history; columns currently include epoch, train_loss, val_loss, val_mae, val_psnr, val_ssim, learning_rate)

eval_results.json schema v1 (masked-only, CelebA-focused)
- timestamp: ISO 8601 run time
- evaluated_count: integer number of test samples evaluated
- metrics_masked:
  - psnr: float
  - ssim: float
  - mae: float
  - lpips: float (optional)

Assumptions and notes
- Unmasked portions are constant; only masked-region metrics are reported.
- For masked PSNR/SSIM utilities, see [flowmatching/training/metrics.py](flowmatching/training/metrics.py); optional LPIPS/FID utilities live in [evaluation/metrics.py](evaluation/metrics.py).

Quick references
- Shared config: [common_config.py](common_config.py)
- FlowMatching pipeline: [flowmatching/pipeline.py](flowmatching/pipeline.py)
- VAE pipeline: [vae/pipeline.py](vae/pipeline.py)
- Diffusion pipeline (stub): [diffusion/pipeline.py](diffusion/pipeline.py)
- Metrics (masked PSNR/SSIM): [flowmatching/training/metrics.py](flowmatching/training/metrics.py)
- Metrics (LPIPS/FID helpers): [evaluation/metrics.py](evaluation/metrics.py)
