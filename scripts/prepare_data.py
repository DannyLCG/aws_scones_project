"""
Rebuild the .lst manifests SageMaker's image-classification format needs

Why this script has to exist: data/train and data/test hold the extracted
PNGs, but their filenames are CIFAR's descriptive names, not class names;
i.e., "bike_*" shows up under both fine labels So grouping by
filename prefix silently mislabels images. The only authoritative source is
CIFAR-100's own fine_labels

Run this once; the generated data/train.lst and data/test.lst are committed,
so scripts/train.py never needs the CIFAR archive again.

Usage:
    uv run python scripts/prepare_data.py
"""
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE = DATA_DIR / ".cifar-100-python.tar.gz"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

# CIFAR-100 fine labels mapped to the binary labels the model trains on 
# (`0 if x == 8 else 1`)
BICYCLE, MOTORCYCLE = 8, 48
BINARY_LABEL = {BICYCLE: 0, MOTORCYCLE: 1}


def download_cifar() -> Path:
    # Cache the archive next to the data so re-runs are offline
    if CACHE.exists():
        print(f"using cached {CACHE.name}")
        return CACHE
    print(f"downloading CIFAR-100 (~169 MB) from {CIFAR_URL} ...")
    urllib.request.urlretrieve(CIFAR_URL, CACHE)
    return CACHE


def load_filename_labels(archive: Path) -> dict[str, int]:
    """Map every CIFAR filename to its fine label, for the two classes we keep."""
    mapping: dict[str, int] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member_name in ("cifar-100-python/train", "cifar-100-python/test"):
            fh = tar.extractfile(member_name)
            if fh is None:
                sys.exit(f"missing {member_name} inside {archive.name}")
            batch = pickle.load(fh, encoding="bytes")
            filenames = batch[b"filenames"]
            fine_labels = batch[b"fine_labels"]
            for raw_name, label in zip(filenames, fine_labels):
                if label in BINARY_LABEL:
                    mapping[raw_name.decode("utf-8")] = label
    return mapping


def write_manifest(split: str, labels: dict[str, int]) -> None:
    """Emit <row>\t<label>\t<filename>, the .lst layout SageMaker expects."""
    image_dir = DATA_DIR / split
    images = sorted(p.name for p in image_dir.glob("*.png"))
    if not images:
        sys.exit(f"no PNGs found in {image_dir}")

    rows, missing = [], []
    for row, name in enumerate(images):
        fine = labels.get(name)
        if fine is None:
            # A local PNG with no CIFAR label would silently corrupt training
            missing.append(name)
            continue
        rows.append(f"{row}\t{BINARY_LABEL[fine]}\t{name}")

    if missing:
        sys.exit(f"{len(missing)} image(s) in {split} not found in CIFAR: {missing[:5]}")

    out = DATA_DIR / f"{split}.lst"
    out.write_text("\n".join(rows) + "\n")

    counts = {0: 0, 1: 0}
    for r in rows:
        counts[int(r.split("\t")[1])] += 1
    print(f"{out.relative_to(REPO_ROOT)}: {len(rows)} rows "
          f"({counts[0]} bicycle / {counts[1]} motorcycle)")


def main() -> None:
    archive = download_cifar()
    labels = load_filename_labels(archive)
    print(f"CIFAR fine labels loaded for {len(labels)} bicycle/motorcycle images")
    for split in ("train", "test"):
        write_manifest(split, labels)


if __name__ == "__main__":
    main()
