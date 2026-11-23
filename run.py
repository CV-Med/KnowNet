"""KnowNet entrypoint.

Usage
-----
    python run.py --mode train   --config configs/config.yaml
    python run.py --mode test    --config configs/config.yaml
    python run.py --mode eval    --config configs/config.yaml
    python run.py --mode ablation
    python run.py --mode gating
    python run.py --mode smoke

The script changes the working directory to the project root, so every path in
``configs/config.yaml`` is relative to this file.
"""

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from data.dataset import get_datasets
from engines.evaluator import Evaluator
from engines.trainer import Trainer
from models import build_model
from utils.device import resolve_device, device_summary
from utils.logger import Logger
from utils.temperature import calibrate_model
from utils.reproducibility import set_seed


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_dataloaders(config: dict, splits: List[str] = ("train", "val")):
    datasets = get_datasets(config)
    dataset_map = {"train": datasets[0], "val": datasets[1], "test": datasets[2]}
    data_config = config.get("data", {})
    num_workers = int(data_config.get("num_workers", 0))
    batch_size = int(config.get("training", {}).get("batch_size", 2))
    pin_memory = resolve_device(config.get("device", "auto")).type == "cuda"

    loaders = {}
    for split in splits:
        dataset = dataset_map[split]
        if split == "train":
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
                drop_last=len(dataset) > batch_size,
                persistent_workers=num_workers > 0,
            )
        else:
            # Full 3D volumes are evaluated one case at a time.
            loaders[split] = DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=min(2, num_workers), pin_memory=pin_memory,
            )
    return loaders, datasets


def build_recorder(config: dict):
    from utils.experiment_recorder import ExperimentRecorder

    ablation_config = config.get("ablation", {})
    sota_config = config.get("sota", {})
    sota_baselines = {k: float(v) for k, v in sota_config.items() if isinstance(v, (int, float))}
    return ExperimentRecorder(
        innovation_names=ablation_config.get("innovation_names", ["BGate", "PrGate", "EvCal"]),
        sota_baselines=sota_baselines,
        baseline_metrics={},
        iteration_round=config.get("iteration", {}).get("round", 1),
        dataset_name=config.get("data", {}).get("dataset_name", ""),
    )


def load_trained_model(config: dict, checkpoint: Optional[str], device: torch.device,
                       disabled_modules: Optional[List[str]] = None,
                       gating_variant: Optional[str] = None):
    model = build_model(config, disabled_modules=disabled_modules)
    if gating_variant is not None:
        model = build_model({**config, "model": {**config["model"], "gating_variant": gating_variant}})
    model.to(device)

    checkpoint = checkpoint or os.path.join("weights", "best_checkpoint.pth")
    if os.path.exists(checkpoint):
        from utils.checkpoint import load_checkpoint
        missing, unexpected = load_checkpoint(model, checkpoint, device)
        print(
            f"[checkpoint] loaded {checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        print(
            f"[checkpoint] WARNING: {checkpoint} not found; "
            "running with randomly initialised weights. Download the released "
            "checkpoint (see weight/download.txt) for meaningful predictions."
        )
    return model


def prepare_run(config: dict, args) -> torch.device:
    set_seed(config.get("seed", 42), config.get("deterministic", True))
    if args.device:
        config["device"] = args.device
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.round is not None:
        config.setdefault("iteration", {})["round"] = args.round
    device = resolve_device(config.get("device", "auto"))
    print(f"[env] {device_summary(device)} | selected={device}")
    return device


def evaluate_split(config: dict, loader, model, device: torch.device,
                   sliding_window: Optional[bool] = None) -> Dict:
    evaluator = Evaluator(
        model, loader, config, device=device.type,
        sliding_window=sliding_window,
    )
    results = evaluator.evaluate()
    print(evaluator.summary(results))
    return results


def run_train(config: dict, args) -> Dict:
    device = prepare_run(config, args)
    loaders, datasets = build_dataloaders(config, splits=("train", "val", "test"))
    print(f"[data] train={len(datasets[0])} val={len(datasets[1])} test={len(datasets[2])}")
    model = build_model(config)
    model.to(device)
    print(f"[model] parameters={count_parameters(model) / 1e6:.2f}M")

    recorder = build_recorder(config)
    trainer = Trainer(
        model, loaders["train"], loaders["val"], config,
        recorder=recorder, checkpoint_dir=args.checkpoint_dir,
    )
    trainer.fit()
    recorder.save()

    if os.path.exists(os.path.join(args.checkpoint_dir, "best_checkpoint.pth")):
        from utils.checkpoint import load_checkpoint
        load_checkpoint(model, os.path.join(args.checkpoint_dir, "best_checkpoint.pth"), device)
    results = evaluate_split(config, loaders["test"], model, device)
    with open("test_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "per_case"}, f, indent=2)

    # Fit the EvCal temperature on the validation split after training and
    # store it next to the model weights.
    calibrate_model(
        config, model, loaders["val"], device,
        os.path.join(args.checkpoint_dir, "best_checkpoint_calibrated.pth"),
    )
    return results


def run_test(config: dict, args, mode: str = "test") -> Dict:
    device = prepare_run(config, args)
    split = "val" if mode == "eval" else "test"
    loaders, _ = build_dataloaders(config, splits=(split,))
    model = load_trained_model(config, args.checkpoint, device)
    results = evaluate_split(config, loaders[split], model, device)
    with open(f"{split}_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "per_case"}, f, indent=2)
    return results


def run_ablation(config: dict, args) -> List[Dict]:
    """Module ablation: disable one innovation module at a time."""
    device = prepare_run(config, args)
    loaders, datasets = build_dataloaders(config, splits=("train", "val", "test"))
    print(f"[data] train={len(datasets[0])} val={len(datasets[1])} test={len(datasets[2])}")

    ablation_config = config.get("ablation", {})
    finetune_epochs = int(args.epochs or ablation_config.get("finetune_epochs", 30))
    module_names = ablation_config.get("innovation_names", ["BGate", "PrGate", "EvCal"])

    reference_model = load_trained_model(config, args.checkpoint, device)
    reference = evaluate_split(config, loaders["test"], reference_model, device)
    rows = [{"variant": "full", "dice": reference["dice_mean"],
             "hd95": reference.get("hd95_mean")}]
    print(f"[ablation] full        Dice={reference['dice_mean'] * 100:.2f}")

    for module_name in module_names:
        model = load_trained_model(
            config, args.checkpoint, device, disabled_modules=[module_name]
        )
        trainer = Trainer(
            model, loaders["train"], loaders["val"], config,
            checkpoint_dir=args.checkpoint_dir,
        )
        print(f"[ablation] fine-tuning without {module_name} for {finetune_epochs} epochs")
        trainer.fit(epochs=finetune_epochs, load_checkpoint=False)
        results = evaluate_split(config, loaders["test"], model, device)
        rows.append({
            "variant": f"w/o {module_name}",
            "dice": results["dice_mean"],
            "hd95": results.get("hd95_mean"),
        })
        print(f"[ablation] w/o {module_name:<8s} Dice={results['dice_mean'] * 100:.2f}")

    with open("ablation_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_gating(config: dict, args) -> List[Dict]:
    """Gating-design ablation over the five PrGate variants."""
    device = prepare_run(config, args)
    loaders, datasets = build_dataloaders(config, splits=("train", "val", "test"))
    print(f"[data] train={len(datasets[0])} val={len(datasets[1])} test={len(datasets[2])}")

    gating_config = config.get("ablation", {})
    variants = gating_config.get(
        "gating_variants", ["uniform", "global", "coord", "feature", "coord_feature"]
    )
    epochs = int(args.epochs or config.get("training", {}).get("epochs", 200))

    rows = []
    for variant in variants:
        variant_config = copy.deepcopy(config)
        variant_config.setdefault("model", {})["gating_variant"] = variant
        model = build_model(variant_config)
        model.to(device)
        gate_params = sum(
            p.numel() for n, p in model.named_parameters()
            if "prgate.coord_mlp" in n or "prgate.feature_logit" in n or "prgate.global_logits" in n
        )
        print(
            f"[gating] {variant}: gate_params={gate_params} "
            f"total={count_parameters(model) / 1e6:.2f}M, training {epochs} epochs"
        )
        trainer = Trainer(
            model, loaders["train"], loaders["val"], variant_config,
            checkpoint_dir=args.checkpoint_dir,
        )
        trainer.fit(epochs=epochs, load_checkpoint=False)
        results = evaluate_split(config, loaders["test"], model, device)
        rows.append({
            "variant": variant,
            "dice": results["dice_mean"],
            "hd95": results.get("hd95_mean"),
            "gate_params": gate_params,
        })
        print(f"[gating] {variant:<14s} Dice={results['dice_mean'] * 100:.2f}")

    with open("gating_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_calibrate(config: dict, args) -> Dict:
    device = prepare_run(config, args)
    loaders, _ = build_dataloaders(config, splits=("val",))
    model = load_trained_model(config, args.checkpoint, device)
    destination = args.output_checkpoint or os.path.join(
        args.checkpoint_dir, "best_checkpoint_calibrated.pth"
    )
    return calibrate_model(config, model, loaders["val"], device, destination)


def run_smoke(config: dict, args) -> None:
    """Dependency/data-free self-check for CI and CPU-only workstations."""
    device = prepare_run(config, args)
    logger = Logger(log_dir="logs", name="smoke")

    # 1. Forward/backward on a small synthetic volume.
    model = build_model(config)
    model.to(device)
    model.train()
    volume = torch.randn(1, 4, 32, 32, 32, device=device)
    target = torch.randint(0, 4, (1, 32, 32, 32), device=device)
    logits, aux, uncertainty = model(volume)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    for aux_logit in aux:
        if tuple(aux_logit.shape[2:]) != tuple(target.shape[1:]):
            aux_logit = torch.nn.functional.interpolate(
                aux_logit, size=target.shape[1:], mode="trilinear", align_corners=False
            )
        loss = loss + 0.1 * criterion(aux_logit, target)
    loss.backward()
    logger.info(
        f"forward/backward OK: logits={tuple(logits.shape)} "
        f"aux={len(aux)} U={float(uncertainty.detach().mean()):.3f} "
        f"loss={float(loss.detach()):.4f}"
    )

    # 2. Every PrGate gating variant must build and run.
    for variant in config.get("ablation", {}).get(
        "gating_variants", ["uniform", "global", "coord", "feature", "coord_feature"]
    ):
        variant_config = copy.deepcopy(config)
        variant_config["model"]["gating_variant"] = variant
        variant_model = build_model(variant_config).to(device).eval()
        with torch.no_grad():
            out = variant_model(torch.randn(1, 4, 32, 32, 32, device=device))[0]
        assert out.shape == (1, 4, 32, 32, 32), (variant, out.shape)
        logger.info(f"gating variant {variant}: OK")

    # 3. Sliding-window inference over a volume larger than the window.  A
    #    narrow variant of the network is used so this stays cheap on CPU.
    from utils.inference import sliding_window_logits
    small_config = copy.deepcopy(config)
    small_config["model"]["enc_features"] = [8, 16, 32, 64, 96]
    small_config["model"]["bottleneck_channels"] = 96
    small_model = build_model(small_config).to(device).eval()
    with torch.no_grad():
        sw = sliding_window_logits(
            small_model, torch.randn(1, 4, 64, 64, 64, device=device),
            window_size=(32, 32, 32), stride=(32, 32, 32),
        )
    assert sw.shape == (1, 4, 64, 64, 64), sw.shape
    logger.info(f"sliding-window inference OK: {tuple(sw.shape)}")

    # 4. Temperature fitting on synthetic logits (tumour voxels only).
    from utils.calibration import fit_temperature_brier
    logits = torch.randn(2, 4, 8, 8, 8, device=device)
    labels = torch.randint(1, 4, (2, 8, 8, 8), device=device)
    tau, info = fit_temperature_brier(logits, labels, num=50)
    logger.info(f"temperature fitting OK: tau={tau:.3f} (grid={len(info['tau'])})")
    logger.info(f"parameters: {count_parameters(model) / 1e6:.2f}M | smoke test passed")


def main():
    parser = argparse.ArgumentParser(description="KnowNet training/evaluation entrypoint")
    parser.add_argument("--mode", required=True,
                        choices=["train", "test", "eval", "ablation", "gating",
                                 "calibrate", "smoke"])
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default=None,
                        help="auto | cpu | mps | cuda | cuda:0 (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training epochs (useful for smoke runs)")
    parser.add_argument("--round", type=int, default=None, help="iteration round label")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path (default: weights/best_checkpoint.pth)")
    parser.add_argument("--checkpoint-dir", default="weights")
    parser.add_argument("--output-checkpoint", default=None,
                        help="destination of the calibrated checkpoint")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="repeat training/gating runs with seeds seed+i")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "smoke":
        run_smoke(config, args)
        return

    if args.num_seeds > 1 and args.mode in ("train", "gating"):
        for offset in range(args.num_seeds):
            seed = int(config.get("seed", 42)) + offset
            run_config = copy.deepcopy(config)
            run_config["seed"] = seed
            run_config["data"]["split_seed"] = seed
            print(f"\n===== seed {seed} ({offset + 1}/{args.num_seeds}) =====")
            if args.mode == "train":
                run_train(run_config, args)
            else:
                run_gating(run_config, args)
        return

    if args.mode == "train":
        run_train(config, args)
    elif args.mode == "test":
        run_test(config, args, mode="test")
    elif args.mode == "eval":
        run_test(config, args, mode="eval")
    elif args.mode == "ablation":
        run_ablation(config, args)
    elif args.mode == "gating":
        run_gating(config, args)
    elif args.mode == "calibrate":
        run_calibrate(config, args)


if __name__ == "__main__":
    main()
