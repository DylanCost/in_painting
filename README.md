# CelebA Inpainting

## Requirements

Install requirements via

```
pip install -r requirements.txt
```

## Common Modules

The following modules are shared between models:
- `data/celeba_dataset.py` - CelebA dataset class.
- `evaluation/metrics.py` - Evaluation metrics.
- `common_config.py` - Common configuration for all models.
- `masking/mask_generatory.py` - Mask generation logic

## Flowmatching Pipeline

The flowmatching pipeline can be run via

```
python -m flowmatching.pipeline --epochs 100 --batch_size 64 --num_eval_samples 1024 --num_example_images 8
```

This outputs runs stored in the following directory structure:

```
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
```
