"""Build an Ultralytics layout for FloW-Img (ablation 08).

Official FloW_IMG has training/ (1200) and test/ (800), Pascal VOC, single class
`bottle`. The test split stays the scored split (same images as ablation 04), so a
seeded slice of training/ is held out as val for checkpoint selection.

Output (images are symlinks):
  <out>/{train,val,test}/images/*.jpg
  <out>/{train,val,test}/labels/*.txt
  <out>/data.yaml
"""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_NAMES = {"bottle", "waste", "floating waste", "object"}


def voc_to_yolo_lines(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    img_w = float(root.findtext("size/width"))
    img_h = float(root.findtext("size/height"))
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASS_NAMES:
            raise ValueError(f"Unexpected class {name!r} in {xml_path}")
        bb = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        xmin, xmax = max(0.0, xmin), min(img_w, xmax)
        ymin, ymax = max(0.0, ymin), min(img_h, ymax)
        w, h = xmax - xmin, ymax - ymin
        if w <= 0 or h <= 0:
            continue
        cx, cy = xmin + w / 2, ymin + h / 2
        lines.append(f"0 {cx / img_w:.6f} {cy / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}")
    return lines


def write_split(name: str, stems: list[str], src: Path, out: Path) -> int:
    img_dir, lbl_dir = out / name / "images", out / name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for stem in stems:
        lines = voc_to_yolo_lines(src / "annotations" / f"{stem}.xml")
        n_boxes += len(lines)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        link = img_dir / f"{stem}.jpg"
        if not link.exists():
            link.symlink_to(src / "images" / f"{stem}.jpg")
    print(f"{name}: {len(stems)} images, {n_boxes} boxes")
    return n_boxes


def stems_of(split_dir: Path) -> list[str]:
    stems = sorted(p.stem for p in (split_dir / "images").glob("*.jpg"))
    missing = [s for s in stems if not (split_dir / "annotations" / f"{s}.xml").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images without XML in {split_dir}, e.g. {missing[:3]}")
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-root", type=Path, required=True, help="dir with training/ and test/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_all = stems_of(args.flow_root / "training")
    test = stems_of(args.flow_root / "test")
    rng = random.Random(args.seed)
    shuffled = train_all[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * args.val_frac)
    val, train = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])

    write_split("train", train, args.flow_root / "training", args.out)
    write_split("val", val, args.flow_root / "training", args.out)
    write_split("test", test, args.flow_root / "test", args.out)

    (args.out / "data.yaml").write_text(
        f"path: {args.out}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "nc: 1\nnames: ['waste']\n"
    )
    (args.out / "val_split.txt").write_text("\n".join(val) + "\n")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
