# KnowNet — Brain Tumor Segmentation (BraTS2021)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A pure 3D ConvNet for whole-brain tumor segmentation on the BraTS2021 dataset, built on a **SegResNet** backbone with three pluggable innovation modules:

| Module | Type | Description |
|--------|------|-------------|
| **BoundGATE** | T3 | Multi-scale boundary gradient → Sigmoid gating at skip connection |
| **PyraGATE** | T3→L4 | Voxel-coordinate MLP → softmax-gated tri-dilation (d=1,2,3) DWConv3d fusion at bottleneck |
| **EKD-CALIB** | T3 | Learnable temperature τ calibrated evidential uncertainty |

## Architecture

```
Input (4×D×H×W) → Encoder (5-stage ResBlock3D) → Bottleneck → Decoder (5-stage, deep supervision)

Skip (stage-2) ──► BoundGATE ──► gated feature to decoder
Bottleneck ──────► PyraGATE ───► multi-dilation fused feature
Logits ──────────► EKD-CALIB ──► calibrated uncertainty
```

- **Backbone**: SegResNet-style 3D ResBlock encoder-decoder with GroupNorm & LeakyReLU
- **Deep supervision**: 4 auxiliary outputs with decreasing weights [0.4, 0.3, 0.2, 0.1]
- **Uncertainty**: Evidence-based uncertainty (EKD-CALIB) guides auxiliary loss weighting

## Requirements

- Python 3.9+
- PyTorch ≥ 2.0.0 (CUDA required)
- 16GB+ GPU memory recommended

```bash
pip install -r requirements.txt
```

## Data

1. Download BraTS2021 from [Synapse](https://www.synapse.org/Synapse:syn27097444) or [Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
2. Organize as `data/BraTS2021/BraTS2021_*/` with files:

```
BraTS2021_{id}_flair.nii.gz
BraTS2021_{id}_t1.nii.gz
BraTS2021_{id}_t1ce.nii.gz
BraTS2021_{id}_t2.nii.gz
BraTS2021_{id}_seg.nii.gz
```

Label 4 (enhancing tumor) is remapped to 3 internally. Evaluation regions: WT (Whole Tumor = 1+2+3), TC (Tumor Core = 1+3), ET (Enhancing Tumor = 3).

## Usage

### Train

```bash
python run.py --mode train --config configs/config.yaml
```

- 200 epochs, AdamW (lr=5e-4, wd=1e-4), Poly LR + 5-epoch warmup
- Gradient accumulation (4 steps), dual gradient clipping (0.5)
- Auto-loads `weights/best_checkpoint.pth` if available
- Checkpoints saved to `weights/`

```
python run.py --mode train --config configs/config.yaml --round 3
```

### Evaluate

```bash
python run.py --mode test --config configs/config.yaml
```

Loads `weights/best_checkpoint.pth` and prints Dice / IoU / HD95.

### Ablation Study

```bash
python run.py --mode ablation --config configs/config.yaml
```

Systematically disables each innovation module, finetunes for 30 epochs, records ΔDice and Cohen's d effect size to `experiment_summary.json`.

```bash
python run.py --mode ablation --config configs/config.yaml --num_seeds 5
```

### Batch Inference + Visualization

```bash
python predict/predict.py
```

Processes `predict/BraTS2021_*/` cases and generates PNG overlays (prediction + GT overlay on Flair) in `predict/results/`.

## Results

| Region | KnowNet (Target) | SOTA Baseline |
|--------|------------------|---------------|
| Whole Tumor (WT) | — | 0.927 |
| Tumor Core (TC) | — | 0.891 |
| Enhancing Tumor (ET) | — | 0.860 |
| **Mean Dice** | — | **0.893** |

*SOTA baselines from SegResNet (MONAI 1.2) and TransBTS (MICCAI 2021).*

## Config

All hyperparameters in `configs/config.yaml`:
- Architecture dimensions, dilation rates, coordinate MLP hidden size
- Training schedule, optimizer, scheduler, warmup
- Loss weights, gradient clipping, accumulation steps
- Ablation: module names to disable, finetune epochs
- Optional `--round N` CLI argument overrides iteration round

## Project Structure

```
configs/          # YAML configuration
data/             # Dataset & transforms (BraTSDataset)
engines/          # Trainer & Evaluator
losses/           # Deep supervision Dice loss
models/           # BrainTumorSegModel + backbone + innovation modules
utils/            # Metrics, reproducibility, logger, experiment recorder
predict/          # Batch inference + visualization
weights/          # Checkpoints (gitignored)
run.py            # Entry point
```

## Building Blocks

Each module can be disabled at build time:

```python
model = build_model(config, disabled_modules=["BoundGATE", "PyraGATE", "EKD_CALIB"])
```

## License

This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

## Citation

If you use this code, please cite:

```bibtex
@article{KnowNet2025,
  title={KnowNet: Knowledge-Guided Brain Tumor Segmentation with Boundary Gradient Attention and Evidential Calibration},
  author={},
  journal={Knowledge-Based Systems},
  year={2025}
}
```
