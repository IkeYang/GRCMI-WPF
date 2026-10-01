import argparse
import csv
import datetime
import json
import logging
import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pangu_submitcode.data.loader import create_dataloaders
from pangu_submitcode.data.loader_utils import PanguWeatherDataLoader, WindFarmDataLoader
from pangu_submitcode.models.GRCMI_WPF import GRCMI_WPF


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_mae",
    "train_rmse",
    "train_sde",
    "train_mape",
    "val_loss",
    "val_mae",
    "val_rmse",
    "val_sde",
    "val_mape",
    "test_loss",
    "test_mae",
    "test_rmse",
    "test_sde",
    "test_mape",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRCMI-WPF with Pangu weather input")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("pangu_submitcode", "config", "configGRCMI_WPF_submit.json"),
        help="Path to JSON config.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=os.path.join("pangu_submitcode", "experiments"),
        help="Output root directory.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("train_grcmi_wpf")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def build_output_dir(config: Dict, output_root: str) -> str:
    model_name = config["model"]["name"]
    dataset = config["dataset_name"]
    turbine_indices = config["turbine_indices"]
    sequence_length = config["sequence_length"]
    prediction_horizon = config["prediction_horizon"]
    turbine_tag = "-".join(str(i) for i in turbine_indices)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{model_name}_{dataset}_WT{turbine_tag}_seq{sequence_length}_pred{prediction_horizon}_{timestamp}"
    return os.path.join(output_root, folder_name)


def validate_required_files(config: Dict) -> None:
    data_root = config["data_root"]
    dataset_name = config["dataset_name"]

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    wind_loader = WindFarmDataLoader(data_root)
    if dataset_name not in wind_loader.dataset_specs:
        raise ValueError(f"Unknown dataset_name in config: {dataset_name}")

    dataset_spec = wind_loader.dataset_specs[dataset_name]
    required_wind_files = []
    if "filename" in dataset_spec:
        required_wind_files.append(dataset_spec["filename"])
    if "files" in dataset_spec:
        required_wind_files.extend(dataset_spec["files"])

    for filename in required_wind_files:
        full_path = os.path.join(data_root, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing wind data file: {full_path}")

    upper_file = os.path.join(data_root, f"pangu4{dataset_name}_upper.npy")
    surface_file = os.path.join(data_root, f"pangu4{dataset_name}_surface.npy")
    if not os.path.exists(upper_file):
        raise FileNotFoundError(f"Missing Pangu upper-layer file: {upper_file}")
    if not os.path.exists(surface_file):
        raise FileNotFoundError(f"Missing Pangu surface-layer file: {surface_file}")


def write_metrics_header(csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def append_metrics_row(csv_path: str, row: Dict) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writerow(row)


def forward_batch(model: nn.Module, batch: Tuple, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(batch) != 4:
        raise ValueError("Expected batch to contain (wind_x, upper_weather, surface_weather, y)")

    wind_x, upper_weather, surface_weather, y = batch
    wind_x = wind_x.to(device)
    upper_weather = upper_weather.to(device)
    surface_weather = surface_weather.to(device)
    y = y.to(device).squeeze()

    y_pred = model(wind_x, upper_weather, surface_weather).squeeze()
    return y_pred, y


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> Dict[str, float]:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    diff = pred_flat - target_flat

    loss = criterion(pred, target).item()
    mae = torch.mean(torch.abs(diff)).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    sde = torch.std(diff, unbiased=False).item()

    non_zero_mask = target_flat != 0
    if torch.any(non_zero_mask):
        mape = torch.mean(torch.abs(diff[non_zero_mask] / target_flat[non_zero_mask])).item() * 100.0
    else:
        mape = 0.0

    return {
        "loss": loss,
        "mae": mae,
        "rmse": rmse,
        "sde": sde,
        "mape": mape,
    }


def evaluate_loader(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            y_pred, y = forward_batch(model, batch, device)
            preds.append(y_pred.detach().cpu())
            targets.append(y.detach().cpu())

    all_preds = torch.cat(preds, dim=0)
    all_targets = torch.cat(targets, dim=0)
    return compute_metrics(all_preds, all_targets, criterion)


def log_loader_brief(train_loader, val_loader, test_loader, logger: logging.Logger) -> None:
    logger.info(
        "batches train=%d val=%d test=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    output_dir: str,
    logger: logging.Logger,
) -> Tuple[int, Dict[str, float]]:
    if num_epochs <= 0:
        raise ValueError(f"training.num_epochs must be > 0, got {num_epochs}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metrics_csv = os.path.join(output_dir, "all_epochs_metrics.csv")
    write_metrics_header(metrics_csv)

    best_epoch = -1
    best_val_mae = float("inf")
    best_test_metrics: Dict[str, float] = {}

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            y_pred, y = forward_batch(model, batch, device)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_loader(model, train_loader, device, criterion)
        val_metrics = evaluate_loader(model, val_loader, device, criterion)
        test_metrics = evaluate_loader(model, test_loader, device, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_sde": train_metrics["sde"],
            "train_mape": train_metrics["mape"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_sde": val_metrics["sde"],
            "val_mape": val_metrics["mape"],
            "test_loss": test_metrics["loss"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_sde": test_metrics["sde"],
            "test_mape": test_metrics["mape"],
        }
        append_metrics_row(metrics_csv, row)

        logger.info(
            "Epoch %d/%d | Val MAE: %.6f | Test MAE: %.6f | Best Val MAE: %.6f",
            epoch,
            num_epochs,
            val_metrics["mae"],
            test_metrics["mae"],
            best_val_mae,
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            best_test_metrics = test_metrics
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            logger.info("Saved new best model at epoch %d", epoch)

    return best_epoch, best_test_metrics


def write_best_epoch_result(
    output_dir: str,
    config: Dict,
    best_epoch: int,
    best_test_metrics: Dict[str, float],
) -> None:
    csv_path = os.path.join(output_dir, "best_epoch_test_results.csv")
    fieldnames = [
        "model",
        "dataset",
        "turbine",
        "best_epoch",
        "mae",
        "rmse",
        "sde",
        "mape",
        "loss",
    ]
    row = {
        "model": config["model"]["name"],
        "dataset": config["dataset_name"],
        "turbine": config["turbine_indices"][0],
        "best_epoch": best_epoch,
        "mae": best_test_metrics.get("mae"),
        "rmse": best_test_metrics.get("rmse"),
        "sde": best_test_metrics.get("sde"),
        "mape": best_test_metrics.get("mape"),
        "loss": best_test_metrics.get("loss"),
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if config.get("model", {}).get("name") != "GRCMI-WPF":
        raise ValueError("Only model.name=GRCMI-WPF is supported in pangu_submitcode/train_grcmi_wpf.py")

    output_dir = build_output_dir(config, args.output_root)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(os.path.join(output_dir, "train.log"))
    logger.info("Config path: %s", args.config)
    logger.info("Output directory: %s", output_dir)

    validate_required_files(config)

    with open(os.path.join(output_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    device_num = int(config["deviceNum"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU fallback is disabled.")
    cuda_count = torch.cuda.device_count()
    if device_num < 0 or device_num >= cuda_count:
        raise RuntimeError(f"Invalid CUDA device index {device_num}. Available device count: {cuda_count}")
    torch.cuda.set_device(device_num)
    device = torch.device(f"cuda:{device_num}")
    logger.info("Using device: %s", device)

    data_root = config["data_root"]
    wind_loader = WindFarmDataLoader(data_root)
    pangu_loader = PanguWeatherDataLoader(data_root)

    train_loader, val_loader, test_loader = create_dataloaders(
        wind_loader,
        pangu_loader,
        dataset_name=config["dataset_name"],
        turbine_indices=config["turbine_indices"],
        feature_indices=config["feature_indices"],
        target_feature_indices=config["target_feature_indices"],
        batch_size=config["batch_size"],
        sequence_length=config["sequence_length"],
        prediction_horizon=config["prediction_horizon"],
        use_weather_data=True,
    )

    log_loader_brief(train_loader, val_loader, test_loader, logger)

    model = GRCMI_WPF(config).to(device)
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.001))
    num_epochs = int(config["training"]["num_epochs"])

    best_epoch, best_test_metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        output_dir=output_dir,
        logger=logger,
    )

    write_best_epoch_result(output_dir, config, best_epoch, best_test_metrics)
    logger.info("Training complete.")
    logger.info("Best epoch: %s", best_epoch)
    logger.info("Best test metrics: %s", best_test_metrics)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
