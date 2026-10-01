import argparse
import os
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual packer for PDM outputs (hard-mode reviewer pipeline)."
    )
    parser.add_argument("--dataset_tag", type=str, required=True, help="Dataset tag, e.g. wf1/wf2/lhb_2017_2020")
    parser.add_argument("--upper_raw", type=str, required=True, help="Path to raw upper array (.npy)")
    parser.add_argument("--surface_raw", type=str, required=True, help="Path to raw surface array (.npy)")
    parser.add_argument("--time_labels", type=str, required=True, help="Path to time labels (.npy)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    return parser.parse_args()


def ensure_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def main() -> int:
    args = parse_args()
    ensure_exists(args.upper_raw)
    ensure_exists(args.surface_raw)
    ensure_exists(args.time_labels)

    upper = np.load(args.upper_raw, allow_pickle=True)
    surface = np.load(args.surface_raw, allow_pickle=True)
    labels = np.load(args.time_labels, allow_pickle=True)

    if upper.ndim != 6:
        raise ValueError(f"upper_raw ndim must be 6, got {upper.ndim}")
    if surface.ndim != 5:
        raise ValueError(f"surface_raw ndim must be 5, got {surface.ndim}")
    if upper.shape[0] != surface.shape[0]:
        raise ValueError(f"time axis mismatch: upper={upper.shape[0]} surface={surface.shape[0]}")
    if labels.shape[0] != upper.shape[0]:
        raise ValueError(f"time labels length mismatch: labels={labels.shape[0]} upper={upper.shape[0]}")

    if upper.shape[1:4] != (4, 5, 13):
        raise ValueError(f"upper_raw core shape must be (4,5,13,*,*), got {upper.shape[1:4]}")
    if surface.shape[1:3] != (4, 4):
        raise ValueError(f"surface_raw core shape must be (4,4,*,*), got {surface.shape[1:3]}")

    os.makedirs(args.output_dir, exist_ok=True)

    upper_out = os.path.join(args.output_dir, f"pangu4{args.dataset_tag}_upper.npy")
    surface_out = os.path.join(args.output_dir, f"pangu4{args.dataset_tag}_surface.npy")

    np.save(
        upper_out,
        {
            "data": upper.astype(np.float32),
            "time_labels": labels,
        },
    )
    np.save(
        surface_out,
        {
            "data": surface.astype(np.float32),
            "time_labels": labels,
        },
    )

    print("saved", upper_out)
    print("saved", surface_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
