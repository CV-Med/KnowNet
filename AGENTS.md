# KnowNet: brain tumor segmentation (BraTS, pure Conv3D)

Task: 3D whole-brain multi-modal tumor segmentation.

## Entrypoint

```sh
python run.py --mode {train,test,eval,ablation,gating,calibrate,smoke} \
    --config configs/config.yaml [--device auto]
```

`run.py` chdirs to the project root, so all paths are relative to the project root.
`--device auto` resolves CUDA → MPS → CPU (`utils/device.py`).

## Architecture

`BrainTumorSegModel` = SegResNet backbone (5-stage 3D ResBlock encoder-decoder,
deep supervision at four decoder levels) + 3 pluggable modules:

- `BoundGATE` (BGate): multi-scale Gaussian-gradient boundary signal →
  sigmoid gating at the stage-2 skip.
- `PrGate` (legacy alias `PyraGATE`): coordinate MLP logits **plus** a 1×1×1
  feature logit branch → softmax-gated tri-dilation (d=1,2,3) depthwise Conv3d
  fusion at the bottleneck. Five designs are selectable through
  `model.gating_variant`: `uniform | global | coord | feature | coord_feature`.
- `EvCal` (legacy aliases `EKDCalib`, `EKD_CALIB`): softplus evidence with a
  fixed τ=1 during training; the temperature is fitted after training on the
  validation split by minimizing the Brier score (τ*=0.82 for BraTS 2021).
  `w_U = 1/(1+Ū)` is detached and scales the auxiliary losses.

Modules are pluggable via `build_model(config, disabled_modules=["BGate","PrGate","EvCal"])`;
the accepted names are case-insensitive aliases (see `models/__init__.py`).

## Commands

| Command | Description |
|---|---|
| `python run.py --mode train` | Full training (200 epochs, AdamW, poly LR + warmup), then test evaluation + τ fit |
| `python run.py --mode test` | Evaluate `weights/best_checkpoint.pth` on the held-out test split |
| `python run.py --mode eval` | Evaluate on the validation split |
| `python run.py --mode calibrate` | Fit the EvCal temperature on the validation split |
| `python run.py --mode ablation` | Module ablation, 30-epoch fine-tune per module |
| `python run.py --mode gating` | Gating-design ablation, five variants |
| `python run.py --mode smoke` | Data/checkpoint-free synthetic self-check |
| `python predict/predict.py` | Batch sliding-window inference + PNG overlays |
| `python -m pytest tests -q` | Full test suite (CPU, < 1 min) |

Optional: `--device`, `--epochs`, `--round`, `--num_seeds N` (nested runs with
`seed+i`, used for multi-seed runs).

## Data

- `data/BraTS2021/BraTS2021_*/` with `{case_id}_{modality}.nii.gz`
  (flair/t1/t1ce/t2) + `{case_id}_seg.nii.gz`; same scheme for BraTS 2020/2023.
- Download links and expected layout: `data/README.md`.
- Label 4 is remapped to 3.
- Evaluation regions: WT (1+2+3), TC (1+3), ET (3).
- Case-level 8:1:1 train/val/test split (`data.split_ratio`, `data.split_seed`);
  BraTS 2021 → 1000/125/126, BraTS 2020 → 295/37/37.
- Training: 128³ crops centred on a random tumour voxel; validation/test: full
  volume + sliding-window inference (128³ window, 64³ stride).
- Transforms: random scale → gamma → noise → contrast → flip → per-scan,
  per-channel z-score.

## Training quirks

- **Optimizer**: AdamW (lr=5e-4, wd=1e-4)
- **Scheduler**: Poly LR (power=0.9) with 5-epoch linear warmup
- **Gradient accumulation**: 4 steps (`accum_steps=4`); dual clipping (per-step
  and after accumulation, clip=0.5)
- **Loss**: deep-supervision Dice (main=1.0, aux=[0.4,0.3,0.2,0.1]); auxiliary
  weight `w_U=1/(1+Ū)` detached, or `constant` 0.75 / `none` for ablations
- **Normalization**: GroupNorm (not BatchNorm); activations LeakyReLU(0.01)
- **Checkpoint**: auto-loads `weights/best_checkpoint.pth`; saves best + latest;
  the calibrated checkpoint additionally stores τ
- **Early stop**: patience 30, validation every 10 epochs
- **Tensor format**: `(batch, channels, depth, height, width)`

## Calibration

- Training temperature is fixed at 1.0; no gradient flows into τ.
- After training, `calibrate_model` collects tumour-voxel logits on the
  validation split and minimizes the Brier score over 500 log-spaced τ values in
  [0.05, 5.00]. τ only rescales evidence, so the argmax is unchanged.
- Calibration metrics in `utils/calibration.py`: ECE (10 equal-width bins in
  [0.5, 1]), ACE (10 equal-mass bins), Brier (summed over 4 classes), NLL, and
  AUROC for error detection (no scikit-learn dependency).

## Ablation

- `--mode ablation` loads the full checkpoint, disables one module at a time,
  fine-tunes 30 epochs and records ΔDice to `ablation_results.json` /
  `experiment_summary.json`.
- `--mode gating` trains all five PrGate designs and records Dice/HD95/gate
  parameters to `gating_results.json`; the gate parameter counts (0 / 0.003K /
  0.115K / 0.963K / 1.078K) are asserted in `tests/test_model.py`.

## Dependencies

`torch>=2.0`, `torchvision`, `monai>=1.2.0`, `nibabel`, `scipy`, `numpy`,
`pyyaml`, `tqdm`, `matplotlib`, `pytest`.

On Apple Silicon, 5-D `torch.softmax` on MPS raises
`Invalid memory formatChannelsLast3d`; PrGate therefore uses a manual stable
softmax over the scale dimension. Keep that implementation when editing.
