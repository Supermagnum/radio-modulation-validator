"""Training pipeline for family and order classifiers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from torch.utils.data import DataLoader, TensorDataset

from rmv.constants import (
    CSPB_TO_FAMILY,
    FAMILY_CLASSES,
    HISARMOD_TO_FAMILY,
    ORDER_CLASSES,
    RADIOML_SKIP_FOR_FAMILY,
    RADIOML_TO_FAMILY,
    SYNTHETIC_TO_FAMILY,
)
from rmv.dataset.loader import load_cspb, load_hisarmod, load_radioml
from rmv.dataset.synthetic import load_synthetic
from rmv.model import ResidualCNN
from rmv.types import IQDataset

logger = logging.getLogger(__name__)
train_console = Console(stderr=True)

TrainMode = Literal["family", "order"]


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    radioml: Path | None = None
    hisarmod: Path | None = None
    cspb: Path | None = None
    synthetic: Path | None = None
    cache_dir: Path = Path(".cache")
    output_dir: Path = Path("checkpoints")
    epochs: int = 50
    batch_size: int = 512
    lr: float = 1e-3
    device: str = "auto"
    patience: int = 10
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    mode: TrainMode = "family"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _family_psk_skew_exceeds_threshold(
    train_labels: np.ndarray,
    num_classes: int,
    *,
    ratio_threshold: float = 3.0,
) -> bool:
    """True when PSK training samples exceed any other family by ratio_threshold."""
    if "PSK" not in FAMILY_CLASSES:
        return False
    psk_idx = FAMILY_CLASSES.index("PSK")
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    psk_count = counts[psk_idx]
    others = np.concatenate([counts[:psk_idx], counts[psk_idx + 1 :]])
    max_other = float(np.max(others)) if others.size else 0.0
    if max_other <= 0:
        return False
    return psk_count / max_other > ratio_threshold


def _order_to_family(order: str, source: str) -> str | None:
    if source == "radioml2016":
        result = RADIOML_TO_FAMILY.get(order)
    elif source == "hisarmod":
        result = HISARMOD_TO_FAMILY.get(order)
    elif source == "cspb":
        result = CSPB_TO_FAMILY.get(order)
    elif source == "synthetic":
        result = SYNTHETIC_TO_FAMILY.get(order)
    else:
        result = None

    if result is None:
        logger.warning(
            "Unknown order %r from source %r — sample will be skipped",
            order,
            source,
        )
    return result


def _build_label_arrays(
    datasets: list[IQDataset],
    mode: TrainMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Concatenate datasets and build integer labels for family or order."""
    samples_list: list[np.ndarray] = []
    labels_list: list[int] = []
    snr_list: list[float] = []

    if mode == "family":
        class_names = FAMILY_CLASSES
        name_to_idx = {c: i for i, c in enumerate(class_names)}
    else:
        class_names = ORDER_CLASSES
        name_to_idx = {c: i for i, c in enumerate(class_names)}

    skipped = 0
    for ds in datasets:
        ds_samples: list[np.ndarray] = []
        ds_labels: list[int] = []
        ds_snr: list[float] = []
        for i, order_idx in enumerate(ds.labels):
            order_name = ds.class_names[int(order_idx)]
            if mode == "family":
                if (
                    ds.source == "radioml2016"
                    and order_name in RADIOML_SKIP_FOR_FAMILY
                ):
                    skipped += 1
                    continue
                fam = _order_to_family(order_name, ds.source)
                if fam is None or fam not in name_to_idx:
                    skipped += 1
                    continue
                ds_labels.append(name_to_idx[fam])
            else:
                if order_name not in name_to_idx:
                    name_to_idx[order_name] = len(class_names)
                    class_names.append(order_name)
                ds_labels.append(name_to_idx[order_name])
            ds_samples.append(ds.samples[i])
            ds_snr.append(float(ds.snr_db[i]))

        if ds_samples:
            samples_list.append(np.stack(ds_samples, axis=0))
            labels_list.extend(ds_labels)
            snr_list.extend(ds_snr)

    if skipped:
        logger.info("Skipped %d samples with unknown or excluded family mapping", skipped)

    if not samples_list:
        msg = "No training samples after family mapping (all excluded or unknown)"
        raise ValueError(msg)

    samples = np.concatenate(samples_list, axis=0)
    labels = np.array(labels_list, dtype=np.int64)
    snr = np.array(snr_list, dtype=np.float32)
    return samples, labels, snr, class_names


def _stratified_split(
    labels: np.ndarray,
    snr: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 stratified split indices by class and SNR bin."""
    rng = np.random.default_rng(seed)
    test_ratio = 1.0 - train_ratio - val_ratio
    snr_bins = np.digitize(snr, bins=[-20, -10, 0, 10, 20])
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for cls in np.unique(labels):
        for snr_bin in np.unique(snr_bins):
            mask = (labels == cls) & (snr_bins == snr_bin)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            rng.shuffle(indices)
            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_idx.extend(indices[:n_train].tolist())
            val_idx.extend(indices[n_train : n_train + n_val].tolist())
            test_idx.extend(indices[n_train + n_val :].tolist())

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def _make_loader(
    samples: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    x = torch.from_numpy(samples[indices])
    y = torch.from_numpy(labels[indices])
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _load_datasets(
    config: TrainConfig,
    progress: Progress,
) -> list[IQDataset]:
    datasets: list[IQDataset] = []
    if config.radioml is not None:
        task = progress.add_task("RadioML: loading...", total=None)
        ds = load_radioml(config.radioml, cache_dir=config.cache_dir)
        progress.update(
            task,
            completed=1,
            total=1,
            description=f"RadioML: {len(ds.samples):,} samples",
        )
        datasets.append(ds)
    if config.hisarmod is not None:
        task = progress.add_task("HISARMOD: loading...", total=None)
        ds = load_hisarmod(config.hisarmod, cache_dir=config.cache_dir)
        progress.update(
            task,
            completed=1,
            total=1,
            description=f"HISARMOD: {len(ds.samples):,} samples",
        )
        datasets.append(ds)
    if config.cspb is not None:
        task = progress.add_task("CSPB: loading...", total=None)
        ds = load_cspb(
            config.cspb,
            cache_dir=config.cache_dir,
            progress=progress,
            progress_task=task,
        )
        progress.update(
            task,
            completed=progress.tasks[task].total or 1,
            description=f"CSPB: {len(ds.samples):,} chunks",
        )
        datasets.append(ds)
    if config.synthetic is not None:
        task = progress.add_task("Synthetic: loading...", total=None)
        ds = load_synthetic(config.synthetic)
        progress.update(
            task,
            completed=1,
            total=1,
            description=f"Synthetic: {len(ds.samples):,} samples",
        )
        datasets.append(ds)
    return datasets


def train_model(config: TrainConfig) -> Path:
    """
    Train family or order classifier; save best checkpoint.

    Returns path to best checkpoint .pt file.
    """
    mode_label = "family" if config.mode == "family" else "order"
    train_console.print(
        f"\n[bold cyan]Training {mode_label} classifier[/] "
        f"({config.epochs} epochs max, batch {config.batch_size})"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=train_console,
        transient=False,
    ) as progress:
        datasets = _load_datasets(config, progress)

    if not datasets:
        msg = (
            "At least one dataset path (--radioml, --hisarmod, --cspb, --synthetic) is required"
        )
        raise ValueError(msg)

    train_console.print("[bold]Building label tensors and splits...[/]")
    samples, labels, snr, class_names = _build_label_arrays(datasets, config.mode)
    train_idx, val_idx, test_idx = _stratified_split(
        labels, snr, config.train_ratio, config.val_ratio
    )
    train_console.print(
        f"  samples: {len(labels):,} | train: {len(train_idx):,} | "
        f"val: {len(val_idx):,} | test: {len(test_idx):,} | classes: {len(class_names)}"
    )

    device = _resolve_device(config.device)
    train_console.print(f"  device: [bold]{device}[/]")
    num_classes = len(class_names)
    model = ResidualCNN(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    train_labels = labels[train_idx]
    if config.mode == "family" and _family_psk_skew_exceeds_threshold(
        train_labels, num_classes
    ):
        class_weights = compute_class_weights(train_labels, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        train_console.print(
            "[yellow]PSK family count >3x other families; using weighted CrossEntropyLoss[/]"
        )
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = _make_loader(samples, labels, train_idx, config.batch_size, shuffle=True)
    val_loader = _make_loader(samples, labels, val_idx, config.batch_size, shuffle=False)
    n_train_batches = len(train_loader)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.output_dir / f"best_{config.mode}_classifier.pt"
    meta_path = config.output_dir / f"best_{config.mode}_meta.json"

    best_acc = 0.0
    epochs_no_improve = 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=train_console,
        transient=False,
    ) as epoch_progress:
        epoch_task = epoch_progress.add_task(
            f"{mode_label} epoch 0/{config.epochs}",
            total=config.epochs,
        )

        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            n_batches = 0
            batch_task: TaskID | None = None
            with Progress(
                TextColumn("  "),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=train_console,
                transient=True,
            ) as batch_progress:
                batch_task = batch_progress.add_task(
                    f"epoch {epoch + 1} train",
                    total=n_train_batches,
                )
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += float(loss.item())
                    n_batches += 1
                    batch_progress.update(batch_task, advance=1)

            scheduler.step()
            val_acc = _evaluate(model, val_loader, device)
            mean_loss = running_loss / max(n_batches, 1)
            epoch_progress.update(
                epoch_task,
                advance=1,
                description=f"{mode_label} epoch {epoch + 1}/{config.epochs}",
            )
            train_console.print(
                f"  epoch {epoch + 1:3d}/{config.epochs}  "
                f"loss={mean_loss:.4f}  val_acc={val_acc:.4f}  "
                f"best={best_acc:.4f}"
            )
            logger.info(
                "Epoch %d/%d loss=%.4f val_acc=%.4f",
                epoch + 1,
                config.epochs,
                mean_loss,
                val_acc,
            )

            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "class_names": class_names,
                        "mode": config.mode,
                        "num_classes": num_classes,
                        "val_acc": val_acc,
                    },
                    best_path,
                )
                meta_path.write_text(
                    json.dumps(
                        {
                            "class_names": class_names,
                            "mode": config.mode,
                            "val_acc": val_acc,
                            "epoch": epoch + 1,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                train_console.print(
                    f"    [green]new best[/] val_acc={val_acc:.4f} -> {best_path.name}"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    train_console.print(
                        f"  [yellow]early stop[/] at epoch {epoch + 1} "
                        f"(patience {config.patience})"
                    )
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

    if not best_path.is_file():
        msg = "Training finished without saving a checkpoint"
        raise RuntimeError(msg)
    train_console.print(f"[green]Done[/] {mode_label} best val_acc={best_acc:.4f}")
    return best_path


def _evaluate(
    model: ResidualCNN,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)


def run_training(
    *,
    radioml: Path | None,
    hisarmod: Path | None,
    cspb: Path | None,
    synthetic: Path | None = None,
    cache: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    device: str,
    order_only: bool = False,
) -> tuple[Path, Path]:
    """Train both family and order classifiers; return checkpoint paths."""
    resolved = _resolve_device(device)
    train_console.print("[bold]Radio Modulation Validator - training[/]")
    train_console.print(f"  cache: {cache}")
    train_console.print(f"  output: {output}")
    if radioml is not None:
        train_console.print(f"  radioml: {radioml}")
    if hisarmod is not None:
        train_console.print(f"  hisarmod: {hisarmod}")
    if cspb is not None:
        train_console.print(f"  cspb: {cspb}")
    if synthetic is not None:
        train_console.print(f"  synthetic: {synthetic}")
    train_console.print(f"  device: {resolved}")

    family_ckpt: Path | None = None
    if not order_only:
        family_ckpt = train_model(
            TrainConfig(
                radioml=radioml,
                hisarmod=hisarmod,
                cspb=cspb,
                synthetic=synthetic,
                cache_dir=cache,
                output_dir=output,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                mode="family",
            )
        )
    order_ckpt = train_model(
        TrainConfig(
            radioml=radioml,
            hisarmod=hisarmod,
            cspb=cspb,
            synthetic=synthetic,
            cache_dir=cache,
            output_dir=output,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            mode="order",
        )
    )
    if family_ckpt is None:
        family_ckpt = output / "best_family_classifier.pt"
    return family_ckpt, order_ckpt
