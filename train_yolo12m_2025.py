#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO12m on 2025 images with labels."
    )
    parser.add_argument(
        "--model",
        default="yolo12m.pt",
        help="Model weights or config (e.g., yolo12m.pt).",
    )
    parser.add_argument(
        "--data-root",
        default=(
            "/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/"
            "temporal-analysis/data/delhi_airshed_y_2025_z_17_buf_25m_symlink"
        ),
        help="Dataset root with images/ and labels/ subfolders.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="1", help="GPU id or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Ultralytics project output directory.",
    )
    parser.add_argument(
        "--name",
        default="yolo12maa_2025",
        help="Ultralytics run name.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["CFCBK", "FCBK", "Zigzag"],
        help="Class names in index order.",
    )
    return parser.parse_args()


def write_splits(
    image_dir: Path, split_dir: Path, val_split: float, seed: int
) -> tuple[Path, Path]:
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No .png files found in {image_dir}")

    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_split))
    val = images[:n_val]
    train = images[n_val:]

    split_dir.mkdir(parents=True, exist_ok=True)
    train_txt = split_dir / "train.txt"
    val_txt = split_dir / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train) + "\n")
    val_txt.write_text("\n".join(str(p) for p in val) + "\n")
    return train_txt, val_txt


def write_data_yaml(
    yaml_path: Path, train_txt: Path, val_txt: Path, names: list[str]
) -> None:
    lines = [
        f"train: {train_txt}",
        f"val: {val_txt}",
        f"nc: {len(names)}",
        "names:",
    ]
    lines.extend([f"  - {name}" for name in names])
    yaml_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    image_dir = data_root / "images"
    label_dir = data_root / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    if not label_dir.exists():
        raise FileNotFoundError(label_dir)

    split_dir = data_root / "splits"
    train_txt, val_txt = write_splits(
        image_dir, split_dir, args.val_split, args.seed
    )
    data_yaml = split_dir / "yolo12maa_2025.yaml"
    write_data_yaml(data_yaml, train_txt, val_txt, args.names)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install it with `pip install ultralytics`."
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
