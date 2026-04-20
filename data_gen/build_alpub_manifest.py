#!/usr/bin/env python3
"""Build a training manifest for the extracted ALPUB_v2 corpus.

The manifest preserves real class counts and includes inverse-frequency class
weights so training can compensate for imbalance without forcing all classes to
the same size.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "ALPUB_v2" / "images"
OUTPUT_DIR = ROOT / "data"
OUTPUT_PATH = OUTPUT_DIR / "alpub_v2_manifest.json"


FOLDER_TO_LABEL = {
    "Alpha": "ALPHA",
    "Beta": "BETA",
    "Gamma": "GAMMA",
    "Delta": "DELTA",
    "Epsilon": "EPSILON",
    "Zeta": "ZETA",
    "Eta": "ETA",
    "Theta": "THETA",
    "Iota": "IOTA",
    "Kappa": "KAPPA",
    "Lambda": "LAMBDA",
    "Mu": "MU",
    "Nu": "NU",
    "Xi": "XI",
    "Omicron": "OMICRON",
    "Pi": "PI",
    "Rho": "RHO",
    "LunateSigma": "LUNATE_SIGMA",
    "Tau": "TAU",
    "Upsilon": "UPSILON",
    "Phi": "PHI",
    "Chi": "CHI",
    "Psi": "PSI",
    "Omega": "OMEGA",
}


LABEL_TO_INDEX = {
    "ALPHA": 0,
    "BETA": 1,
    "GAMMA": 2,
    "DELTA": 3,
    "EPSILON": 4,
    "ZETA": 5,
    "ETA": 6,
    "THETA": 7,
    "IOTA": 8,
    "KAPPA": 9,
    "LAMBDA": 10,
    "MU": 11,
    "NU": 12,
    "XI": 13,
    "OMICRON": 14,
    "PI": 15,
    "RHO": 16,
    "LUNATE_SIGMA": 17,
    "TAU": 18,
    "UPSILON": 19,
    "PHI": 20,
    "CHI": 21,
    "PSI": 22,
    "OMEGA": 23,
}


def parse_document_id(filename: str) -> str:
    """Extract a stable document-ish grouping key from an ALPUB filename."""
    parts = filename.split("_")
    if len(parts) < 2:
        return filename
    return "_".join(parts[:2])


def build_manifest() -> dict:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing source directory: {SOURCE_DIR}")

    records = []
    counts = Counter()
    document_counts = defaultdict(set)

    for folder in sorted(SOURCE_DIR.iterdir()):
        if not folder.is_dir():
            continue

        canonical_label = FOLDER_TO_LABEL.get(folder.name)
        if canonical_label is None:
            raise ValueError(f"Unmapped folder name: {folder.name}")

        label_index = LABEL_TO_INDEX[canonical_label]
        for image_path in sorted(folder.glob("*.jpg")):
            rel_path = image_path.relative_to(ROOT).as_posix()
            document_id = parse_document_id(image_path.name)
            counts[canonical_label] += 1
            document_counts[canonical_label].add(document_id)
            records.append(
                {
                    "path": rel_path,
                    "folder_label": folder.name,
                    "label": canonical_label,
                    "label_index": label_index,
                    "document_id": document_id,
                    "filename": image_path.name,
                }
            )

    total_samples = len(records)
    num_classes = len(LABEL_TO_INDEX)
    class_weights = {}
    class_counts = {}
    class_document_counts = {}
    for label, index in LABEL_TO_INDEX.items():
        count = counts[label]
        class_counts[label] = count
        class_document_counts[label] = len(document_counts[label])
        if count == 0:
            class_weights[label] = 0.0
        else:
            class_weights[label] = total_samples / (num_classes * count)

    return {
        "source": str(SOURCE_DIR.relative_to(ROOT)),
        "num_samples": total_samples,
        "num_classes": num_classes,
        "label_to_index": LABEL_TO_INDEX,
        "folder_to_label": FOLDER_TO_LABEL,
        "class_counts": class_counts,
        "class_document_counts": class_document_counts,
        "class_weights": class_weights,
        "records": records,
    }


def main() -> None:
    manifest = build_manifest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Samples: {manifest['num_samples']}")
    for label, count in manifest["class_counts"].items():
        print(f"{label}: {count} weight={manifest['class_weights'][label]:.4f}")


if __name__ == "__main__":
    main()
