"""Download VisDrone DET; produce a reproducible five-class vehicle dataset."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["motorcycle", "car", "truck", "bus", "other"]
MAPPING = {10: 0, 4: 1, 6: 2, 9: 3, 3: 4, 5: 4, 7: 4, 8: 4}
SOURCE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    """Keep complete files only; a interrupted download can safely be retried."""
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Vehicle-project/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as file:
        size = int(response.headers.get("Content-Length", 0))
        done = 0
        next_log = 0
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
            done += len(chunk)
            if done >= next_log:
                print(f"{path.name}: {done // 1024**2}/{size // 1024**2} MB", flush=True)
                next_log += 100 * 1024**2
        if size and size != done:
            raise IOError(f"Incomplete download: {path.name}")
    if path.suffix == ".zip":
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise IOError(f"Corrupt archive: {path.name}")
    temporary.replace(path)


def convert(rows, width, height):
    labels, ignored = [], []
    for row in rows:
        values = [int(value) for value in row[:8]]
        x, y, w, h, score, category, _, _ = values
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        if score == 0 or category in (0, 11):
            ignored.append((x1, y1, x2, y2))
        elif category in MAPPING:
            labels.append((MAPPING[category], (x1 + x2) / (2 * width),
                           (y1 + y2) / (2 * height), (x2 - x1) / width,
                           (y2 - y1) / height))
    return labels, ignored


def prepare(train_limit=0):
    raw = ROOT / "data/raw"
    target = ROOT / "data/vehicles"
    raw.mkdir(parents=True, exist_ok=True)
    if (target / "manifest.json").exists():
        raise SystemExit("Dataset already prepared. Keep its manifest; use a new checkout to rebuild.")
    manifest = {"dataset": "VisDrone2019-DET vehicles (five classes)", "seed": 32,
                "source": "https://github.com/VisDrone/VisDrone-Dataset",
                "mirror": SOURCE, "mapping": MAPPING, "names": NAMES,
                "protocol": "official train/val/test-dev, ignored regions masked gray; not official VisDrone metrics",
                "splits": {}, "archives": {}, "files": []}
    for split, source_split in [("train", "train"), ("val", "val"), ("test", "test-dev")]:
        name = f"VisDrone2019-DET-{source_split}"
        archive_path = raw / f"{name}.zip"
        download(f"{SOURCE}/{name}.zip", archive_path)
        manifest["archives"][name] = {"sha256": sha256(archive_path), "bytes": archive_path.stat().st_size}
        source_dir = raw / name
        if not source_dir.exists():
            with zipfile.ZipFile(archive_path) as archive:
                for entry in archive.infolist():
                    resolved = (raw / entry.filename).resolve()
                    if not resolved.is_relative_to(raw.resolve()):
                        raise ValueError("Unsafe archive path")
                archive.extractall(raw)
        files = sorted((source_dir / "images").glob("*.jpg"))
        if not files:
            raise ValueError(f"No images in {source_dir}")
        if split == "train" and train_limit:
            # ponytail: deterministic image subset; use full official training split for final training.
            files = sorted(random.Random(32).sample(files, min(train_limit, len(files))))
        image_dir, label_dir = target / "images" / split, target / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        empty = masked = 0
        for image_path in files:
            annotation_path = source_dir / "annotations" / (image_path.stem + ".txt")
            with annotation_path.open(encoding="utf-8") as file:
                rows = [row for row in csv.reader(file) if row]
            with Image.open(image_path) as image:
                labels, ignored = convert(rows, *image.size)
                if ignored:
                    draw = ImageDraw.Draw(image)
                    for box in ignored:
                        draw.rectangle(box, fill=(114, 114, 114))
                    image.save(image_dir / image_path.name, quality=95)
                    masked += 1
                else:
                    shutil.copyfile(image_path, image_dir / image_path.name)
            text = "".join(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for cls, x, y, w, h in labels)
            label_path = label_dir / (image_path.stem + ".txt")
            label_path.write_text(text, encoding="utf-8")
            counts.update(NAMES[label[0]] for label in labels)
            empty += not labels
            manifest["files"].append({"split": split, "image": image_path.name,
                                      "image_sha256": sha256(image_dir / image_path.name),
                                      "label_sha256": sha256(label_path)})
        manifest["splits"][split] = {"images": len(files), "boxes": dict(counts),
                                      "empty_images": empty, "masked_images": masked}
        print(split, manifest["splits"][split], flush=True)
    # The selected official partitions must have no image content overlap.
    seen = {}
    for item in manifest["files"]:
        key = item["image_sha256"]
        if key in seen and seen[key] != item["split"]:
            raise ValueError("Image leakage between splits")
        seen[key] = item["split"]
    target.mkdir(exist_ok=True)
    (target / "data.yaml").write_text(
        f"path: {target.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        + "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES)), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "dataset.json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "files"},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(target / "manifest.json", ROOT / "data/manifest.json")
    print("Prepared", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-limit", type=int, default=0, help="0 = full official training split")
    args = parser.parse_args()
    if args.train_limit < 0:
        parser.error("--train-limit must be non-negative")
    prepare(args.train_limit)
