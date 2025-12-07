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

The flowmatching implementation is contained under `flowmatching/`. The flowmatching pipeline can be run via

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

## Diffusion Pipeline

The diffusion implementation is under `diffusion/`, and you can run the pipeline with

```
python diffusion/pipeline.py
```

Outputs for diffusion are as follows:

```
├─ runs/
│  └─ diffusion/
│     ├─ best_diffusion_metrics.txt
│     ├─ best_test_metrics.txt
│     ├─ diffusion_data.csv
│     ├─ diffusion_training_log.txt
│     ├─ diffusion_testing_log.txt
│     ├─ checkpoints/
│     │  ├─ diffusion_best_model.pt
│     │  └─ diffusion_final_model.pt
│     └─ examples/
│        ├─ samples.png
│        └─ samples_epoch_X
```

### Metrics and Output Files for Diffusion

**`best_diffusion_metrics.txt`**  
Contains the highest validation PSNR, SSIM, and MAE achieved during training.

**`best_test_metrics.txt`**  
Records the batch number and metrics for the best-performing batch during test set evaluation.

**`diffusion_data.csv`**  
Epoch-by-epoch validation metrics (PSNR, SSIM, MAE) in CSV format for DataFrame analysis.

**`diffusion_training_log.txt`**  
Per-epoch training log containing training loss, validation statistics (PSNR, SSIM, MSE, MAE), and the highest PSNR achieved so far.

**`diffusion_testing_log.txt`**  
Per-batch statistics during test evaluation, with average metrics across all batches at the end of the file.

**`checkpoints/diffusion_best_model.pt`**  
Model checkpoint from the epoch with the highest validation PSNR.

**`checkpoints/diffusion_final_model.pt`**  
Model checkpoint from the final training epoch.

**`examples/samples/`**  
Qualitative visualization of the first 8 images from the first test batch.

**`examples/samples_epoch_X/`**  
Training progress visualization showing the same 8 images at epoch 1 and every 20 epochs thereafter.
