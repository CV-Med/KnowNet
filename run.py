import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import BraTSDataset
from data.transforms import get_train_transforms, get_val_transforms
from models import build_model
from engines.trainer import Trainer
from engines.evaluator import Evaluator
from utils.reproducibility import set_seed
from utils.experiment_recorder import ExperimentRecorder


_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)

TASK_PRIMARY_METRIC = {
    "segmentation": "dice",
    "classification": "accuracy",
    "detection": "map",
}


def get_dataloaders(config):
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    train_dataset = BraTSDataset(
        data_root=config["data"]["data_root"],
        modalities=config["data"]["modalities"],
        mode="train",
        crop_size=tuple(config["data"]["crop_size"]),
        train_ratio=config["data"]["train_ratio"],
        transform=train_transform,
    )
    val_dataset = BraTSDataset(
        data_root=config["data"]["data_root"],
        modalities=config["data"]["modalities"],
        mode="val",
        crop_size=tuple(config["data"]["crop_size"]),
        train_ratio=config["data"]["train_ratio"],
        transform=val_transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
        pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
        pin_memory=True, persistent_workers=True,
    )
    return train_loader, val_loader


def run_ablation(config, innovation_names, sota, round_num, num_seeds=1):
    metric_name = TASK_PRIMARY_METRIC.get(config.get("task_type", "segmentation"), "dice")
    train_loader, val_loader = get_dataloaders(config)
    finetune_epochs = config.get("ablation", {}).get("finetune_epochs", 30)

    set_seed(config.get("seed", 42), config.get("deterministic", True))

    full_ckpt_path = "weights/best_checkpoint.pth"
    full_ckpt = torch.load(full_ckpt_path, map_location="cuda")

    model_full = build_model(config, disabled_modules=None)
    model_full = model_full.cuda()
    model_full.load_state_dict(full_ckpt)
    trainer = Trainer(model_full, train_loader, val_loader, config)
    val_results = trainer.validate()
    full_dice = val_results["dice"]
    print(f"Full model  | Dice: {full_dice:.4f}  WT:{val_results['dice_WT']:.3f} TC:{val_results['dice_TC']:.3f} ET:{val_results['dice_ET']:.3f}")

    import tempfile, shutil
    ablated_best = {}
    for name in innovation_names:
        print(f"\n{'='*60}")
        print(f"Ablating: {name}")
        print(f"{'='*60}")
        tmp_dir = tempfile.mkdtemp(prefix=f"ablation_{name}_")

        model_abl = build_model(config, disabled_modules=[name])
        model_abl = model_abl.cuda()
        model_abl.load_state_dict(full_ckpt, strict=False)

        trainer_abl = Trainer(model_abl, train_loader, val_loader, config,
                              checkpoint_dir=tmp_dir)
        trainer_abl.fit(epochs=finetune_epochs, load_checkpoint=False)
        ablated_best[name] = trainer_abl.best_metric
        print(f"  {name} ablated | Best Dice: {ablated_best[name]:.4f} "
              f"(gap: {full_dice - ablated_best[name]:.4f})")

        shutil.rmtree(tmp_dir)

    recorder = ExperimentRecorder(
        innovation_names, sota, {"baseline_dice": full_dice}, round_num,
        dataset_name=config.get("data", {}).get("dataset_name", ""),
    )
    recorder.record_metric(f"best_{metric_name}", full_dice)
    for name in innovation_names:
        recorder.record_ablation_batch(
            name,
            with_scores=[full_dice],
            without_scores=[ablated_best[name]],
            metric_name=metric_name,
            claimed_improvement="",
        )
    recorder.save()
    print(f"\nAblation results saved to experiment_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Segmentation — Final Round (Pure Conv3D)")
    parser.add_argument("--mode", type=str, choices=["train", "test", "eval", "ablation"], default="train")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--num_seeds", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.round is not None:
        config["iteration"]["round"] = args.round

    set_seed(config.get("seed", 42), config.get("deterministic", True))

    if args.mode == "ablation":
        num_seeds = args.num_seeds if args.num_seeds else config.get("experiment", {}).get("num_seeds", 3)
        run_ablation(
            config,
            config["ablation"]["innovation_names"],
            config.get("sota", {}),
            config["iteration"]["round"],
            num_seeds=num_seeds,
        )
        return

    train_loader, val_loader = get_dataloaders(config)

    if args.mode == "train":
        model = build_model(config)
        model = model.cuda()
        recorder = ExperimentRecorder(
            config["ablation"]["innovation_names"],
            config.get("sota", {}),
            {},
            config["iteration"]["round"],
            dataset_name=config.get("data", {}).get("dataset_name", ""),
        )
        trainer = Trainer(model, train_loader, val_loader, config, recorder=recorder)
        trainer.fit()

    elif args.mode in ("test", "eval"):
        model = build_model(config)
        model = model.cuda()
        if os.path.exists("weights/best_checkpoint.pth"):
            state = torch.load("weights/best_checkpoint.pth", map_location="cuda")
            model.load_state_dict(state)
            print(f"Loaded checkpoint from weights/best_checkpoint.pth")
        evaluator = Evaluator(model, val_loader, config, num_classes=config["data"]["num_classes"])
        results = evaluator.evaluate()
        print("=" * 50)
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
