"""
Entry point for training segmentation models on BraTS 2021.

Usage:
    python train.py --config configs/example.yaml

The config file specifies dataset paths, model choice, training hyperparameters,
and run metadata.
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data.dataloader import build_dataloaders
from src.models.unet import build_unet
from src.models.transunet_tiny import build_transunet_tiny
from src.training.trainer import Trainer, set_seed
from src.evaluation.inference import run_inference
from src.evaluation.metrics import compute_metrics


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_name: str, in_channels: int, out_channels: int) -> torch.nn.Module:
    if model_name == "unet":
        return build_unet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "transunet_tiny":
        return build_transunet_tiny(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1. Reproducibility
    set_seed(cfg["seed"])

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Data
    train_loader, val_loader, test_loader = build_dataloaders(
        tar_path=cfg["data"]["tar_path"],
        extracted_dir=cfg["data"]["extracted_dir"],
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
        num_classes=cfg["model"]["num_classes"],
        patch_size=tuple(cfg["data"]["patch_size"]),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        cache_rate=cfg["data"]["cache_rate"],
        seed=cfg["seed"],
        train_fraction=cfg["data"]["train_fraction"],
    )

    # 4. Model
    out_channels = cfg["model"]["num_classes"] + 1 if cfg["model"]["num_classes"] > 1 else 1
    model = build_model(
        model_name=cfg["model"]["name"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=out_channels,
    )

    # 5. Optional W&B init
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["run_name"],
            config=cfg,
        )

    # 6. Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=cfg["model"]["num_classes"],
        learning_rate=cfg["training"]["learning_rate"],
        max_epochs=cfg["training"]["max_epochs"],
        patch_size=tuple(cfg["data"]["patch_size"]),
        sw_batch_size=cfg["training"]["sw_batch_size"],
        output_dir=cfg["output_dir"],
        run_name=cfg["run_name"],
        use_wandb=use_wandb,
    )
    summary = trainer.run()

    # 7. Test evaluation (using best checkpoint)
    print("\nLoading best checkpoint for test evaluation...")
    ckpt_path = Path(cfg["output_dir"]) / "checkpoints" / f"{cfg['run_name']}_best.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        test_results = run_inference(
            model=model,
            data_loader=test_loader,
            device=device,
            num_classes=cfg["model"]["num_classes"],
            patch_size=tuple(cfg["data"]["patch_size"]),
            sw_batch_size=cfg["training"]["sw_batch_size"],
        )

        test_metrics = compute_metrics(
            results=test_results,
            num_classes=cfg["model"]["num_classes"],
            include_hausdorff=cfg.get("evaluation", {}).get("include_hausdorff", False),
        )

        print(f"\nTest results:")
        print(f"  Mean Dice: {test_metrics['mean_dice']:.4f}")
        print(f"  Mean IoU:  {test_metrics['mean_iou']:.4f}")
        if "mean_hausdorff_95" in test_metrics:
            print(f"  Mean HD95: {test_metrics['mean_hausdorff_95']:.4f}")

        if use_wandb:
            wandb.log({"test/" + k: v for k, v in test_metrics.items()
                       if isinstance(v, float)})
    else:
        print("No best checkpoint found — skipping test evaluation.")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()