# KnowNet — Brain Tumor Segmentation (BraTS2021, Pure Conv3D)

Target: Knowledge-Based Systems (KBS). Task: 3D whole-brain tumor segmentation.

## Entrypoint

```sh
python run.py --mode {train,test,eval,ablation} --config configs/config.yaml
```

`run.py` chdirs to project root, so all paths are relative to the project root.

## Architecture

`BrainTumorSegModel` = SegResNet backbone (5-stage 3D ResBlock encoder-decoder) + 3 pluggable innovation modules:

- `BoundGATE` — multi-scale boundary gradient → sigmoid gating at stage-2 skip
- `PyraGATE` — voxel-coord MLP → softmax-gated tri-dilation (d=1,2,3) depthwise Conv3d fusion at bottleneck
- `EKD-CALIB` — learnable temperature τ calibrated evidential uncertainty

Pluggable via `disabled_modules=["BoundGATE","PyraGATE","EKD_CALIB"]` at build time.

## Commands

| Command | Description |
|---|---|
| `python run.py --mode train` | Full training (200 epochs, AdamW, poly LR+warmup) |
| `python run.py --mode test` | Evaluate best checkpoint on val split |
| `python run.py --mode ablation` | Systematic module ablation (30-epoch finetune per module) |
| `python predict/predict.py` | Batch inference + PNG overlays for `predict/BraTS2021_*/` |

Optional: `--round N` overrides `config.yaml` iteration round; `--num_seeds N` for multi-seed ablation.

## Data

- BraTS2021: `data/BraTS2021/BraTS2021_*/` with `{case_id}_{modality}.nii.gz` (flair/t1/t1ce/t2) + `{case_id}_seg.nii.gz`
- Label 4 is remapped to 3 (merges class 4 into 3).
- Evaluation regions: WT (1+2+3), TC (1+3), ET (3).
- 80/20 train/val split per config.
- Transform pipeline: random scale → gamma → noise → contrast → flip → per-channel z-score.

## Training quirks

- **Optimizer**: AdamW (lr=5e-4, wd=1e-4)
- **Scheduler**: Poly LR (power=0.9) with 5-epoch linear warmup
- **Gradient accumulation**: 4 steps (`accum_steps=4`); dual gradient clipping (per-step + accum, clip=0.5)
- **Loss**: Deep supervision Dice loss (main=1.0, aux=[0.4, 0.3, 0.2, 0.1])
- **Normalization**: GroupNorm (not BatchNorm); activations: LeakyReLU(0.01)
- **Checkpoint**: Auto-loads `weights/best_checkpoint.pth` on fit; saves best + latest.
- **Early stop**: patience=30 epochs, validation every 10.
- **Tensor format**: 3D volumes as `(batch, channels, depth, height, width)`.

## Ablation

`--mode ablation` loads the full model checkpoint, builds models with each module disabled one-by-one, finetunes for 30 epochs, and records ΔDice + Cohen's d to `experiment_summary.json`.

## Dependencies

`torch>=2.0`, `torchvision`, `monai>=1.2.0`, `nibabel`, `scipy`, `numpy`, `pyyaml`, `tqdm`, `matplotlib`
