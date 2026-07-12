#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont, UnidentifiedImageError

REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"a": DRAWING_NS, "r": REL_NS}

@dataclass
class ImageRecord:
    global_ordinal: int
    source_docx: str
    source_ordinal: int
    relationship_id: str
    media_path: str
    extracted_file: str
    detected_format: str
    width_px: int
    height_px: int
    mode: str
    bytes: int
    sha256: str
    duplicate_group: str
    analysis_file: str
    thumbnail_file: str
    manuscript_left_page: str = ""
    manuscript_right_page: str = ""
    date_begin: str = ""
    date_end: str = ""
    record_begin: str = ""
    record_end: str = ""
    year_or_transition: str = ""
    hand_or_priest_notes: str = ""
    condition_notes: str = ""
    content_qc_status: str = "not_reviewed"
    confidence: str = ""
    research_notes: str = ""

def natural_key(s: str):
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", s)]

def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def load_config(path: Path) -> dict:
    base = {
        "collection_id": "HTH-0001",
        "collection_title": "San Fernando Baptism Register",
        "global_start": 1,
        "derive": {"grayscale": True, "autocontrast_cutoff": 0.5, "sharpen_factor": 1.0},
        "thumbnails": {"max_width": 420, "max_height": 320, "jpeg_quality": 88},
        "contact_sheets": {"columns": 4, "rows": 5, "cell_width": 480, "cell_height": 380, "margin": 20},
        "source_overrides": {},
    }
    if path.exists():
        return deep_merge(base, json.loads(path.read_text(encoding="utf-8")))
    return base

def discover_docx(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.docx"), key=lambda p: natural_key(p.name))
    return files

def rel_map(z: zipfile.ZipFile) -> dict[str, str]:
    root = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    out = {}
    for rel in root.findall(f"{{{PKG_REL_NS}}}Relationship"):
        rid, target = rel.attrib.get("Id"), rel.attrib.get("Target")
        if rid and target:
            out[rid] = target.lstrip("/") if target.startswith("/") else (Path("word") / target).as_posix()
    return out

def ordered_images(docx: Path):
    with zipfile.ZipFile(docx) as z:
        rels = rel_map(z)
        root = ET.fromstring(z.read("word/document.xml"))
        rids = []
        for blip in root.findall(".//a:blip", NS):
            rid = blip.attrib.get(f"{{{REL_NS}}}embed")
            if rid:
                rids.append(rid)
        for rid in rids:
            media = rels.get(rid)
            if media and media in z.namelist():
                yield rid, media, z.read(media)

def image_info(data: bytes):
    try:
        with Image.open(BytesIO(data)) as im:
            im.load()
            return (im.format or "UNKNOWN").upper(), im.width, im.height, im.mode
    except UnidentifiedImageError as exc:
        raise ValueError("Unsupported embedded image") from exc

def extension(fmt: str) -> str:
    return {"PNG": ".png", "JPEG": ".jpg", "TIFF": ".tif", "BMP": ".bmp", "GIF": ".gif", "WEBP": ".webp"}.get(fmt, ".bin")

def make_analysis(src: Path, dst: Path, cfg: dict):
    with Image.open(src) as im:
        im.load()
        out = im.convert("L") if cfg["derive"]["grayscale"] else im.convert("RGB")
        out = ImageOps.autocontrast(out, cutoff=float(cfg["derive"]["autocontrast_cutoff"]))
        factor = float(cfg["derive"]["sharpen_factor"])
        if factor != 1.0:
            out = ImageEnhance.Sharpness(out).enhance(factor)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, "PNG", optimize=True)

def make_thumb(src: Path, dst: Path, cfg: dict):
    with Image.open(src) as im:
        im.load()
        out = im.convert("RGB")
        out.thumbnail((cfg["thumbnails"]["max_width"], cfg["thumbnails"]["max_height"]), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, "JPEG", quality=cfg["thumbnails"]["jpeg_quality"], optimize=True)

def font():
    try:
        return ImageFont.truetype("arial.ttf", 20)
    except OSError:
        return ImageFont.load_default()

def contact_sheets(records, output: Path, cfg: dict):
    s = cfg["contact_sheets"]
    cols, rows = int(s["columns"]), int(s["rows"])
    per_sheet = cols * rows
    for sheet_no, start in enumerate(range(0, len(records), per_sheet), 1):
        subset = records[start:start + per_sheet]
        canvas = Image.new("RGB", (cols*s["cell_width"] + (cols+1)*s["margin"], rows*s["cell_height"] + (rows+1)*s["margin"]), "white")
        draw = ImageDraw.Draw(canvas)
        for i, rec in enumerate(subset):
            r, c = divmod(i, cols)
            x = s["margin"] + c*(s["cell_width"] + s["margin"])
            y = s["margin"] + r*(s["cell_height"] + s["margin"])
            with Image.open(output / rec.thumbnail_file) as im:
                thumb = im.copy()
            thumb.thumbnail((s["cell_width"], s["cell_height"]-40), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x + (s["cell_width"]-thumb.width)//2, y+35))
            draw.text((x, y+5), f"FS {rec.global_ordinal:04d} | {Path(rec.source_docx).name} #{rec.source_ordinal}", fill="black", font=font())
        d = output / "contact_sheets"
        d.mkdir(parents=True, exist_ok=True)
        canvas.save(d / f"contact_sheet_{sheet_no:03d}.jpg", "JPEG", quality=90, optimize=True)

def write_csv(records, path: Path):
    fields = list(asdict(records[0]).keys()) if records else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))

def preprocess_summary(
    docs_found: int,
    converted: int,
    images: int,
    ocr_count: int,
    errors: int,
) -> None:
    print("\n========== SUMMARY ==========")
    print(f"DOCX found      : {docs_found}")
    print(f"Converted       : {converted}")
    print(f"Images extracted: {images}")
    print(f"OCR performed   : {ocr_count}")
    print(f"Errors          : {errors}")
    print("=============================")
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/source"))
    p.add_argument("--output", type=Path, default=Path("build/preprocessed"))
    p.add_argument("--config", type=Path, default=Path("config/preprocess.json"))
    p.add_argument("--derive", action="store_true")
    p.add_argument("--contact-sheets", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    docs_found = 0
    converted = 0
    images = 0
    ocr_count = 0
    errors = 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)
    docs = discover_docx(args.input)

    docs_found = len(docs)

    if not docs:
        print("No DOCX files found beneath", args.input)
        print("This repository currently contains no source collection.")
        print("See the repository README for layout and usage instructions.")

        preprocess_summary(
            docs_found=docs_found,
            converted=converted,
            images=images,
            ocr_count=ocr_count,
            errors=errors,
        )
        logging.info("Done: %d images -> %s", len(records), args.output)
        return 0
    
    # normal preprocessing continues here
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists; use --overwrite")
        shutil.rmtree(args.output)
    for d in ("raw", "thumbnails", "metadata"):
        (args.output / d).mkdir(parents=True, exist_ok=True)
    if args.derive:
        (args.output / "analysis").mkdir(parents=True, exist_ok=True)

    records = []
    next_global = int(cfg["global_start"])
    for doc in docs:
        override = cfg["source_overrides"].get(doc.name, {})
        images = list(ordered_images(doc))
        skip_first = int(override.get("skip_first", 0))
        skip_last = int(override.get("skip_last", 0))
        if "global_start" in override:
            next_global = int(override["global_start"])
        end = len(images)-skip_last if skip_last else len(images)
        chosen = images[skip_first:end]
        logging.info("%s: %d embedded, %d selected", doc.name, len(images), len(chosen))
        for n, (rid, media, data) in enumerate(chosen, start=skip_first+1):
            fmt, w, h, mode = image_info(data)
            raw_rel = Path("raw") / f"fs_{next_global:04d}{extension(fmt)}"
            raw_path = args.output / raw_rel
            raw_path.write_bytes(data)
            analysis_rel = ""
            if args.derive:
                analysis_rel = (Path("analysis") / f"fs_{next_global:04d}_analysis.png").as_posix()
                make_analysis(raw_path, args.output / analysis_rel, cfg)
            thumb_rel = Path("thumbnails") / f"fs_{next_global:04d}.jpg"
            make_thumb(raw_path, args.output / thumb_rel, cfg)
            records.append(ImageRecord(
                global_ordinal=next_global,
                source_docx=doc.name,
                source_ordinal=n,
                relationship_id=rid,
                media_path=media,
                extracted_file=raw_rel.as_posix(),
                detected_format=fmt,
                width_px=w,
                height_px=h,
                mode=mode,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                duplicate_group="",
                analysis_file=analysis_rel,
                thumbnail_file=thumb_rel.as_posix(),
                research_notes=override.get("label", ""),
            ))
            next_global += 1

    groups = defaultdict(list)
    for r in records:
        groups[r.sha256].append(r)
    dup_no = 0
    for group in groups.values():
        if len(group) > 1:
            dup_no += 1
            label = f"DUP-{dup_no:04d}"
            for r in group:
                r.duplicate_group = label

    meta = args.output / "metadata"
    write_csv(records, meta / "image_manifest.csv")
    write_csv(records, meta / "page_map_template.csv")
    (meta / "image_manifest.json").write_text(json.dumps({
        "schema_version": "0.1",
        "collection_id": cfg["collection_id"],
        "collection_title": cfg["collection_title"],
        "image_count": len(records),
        "records": [asdict(r) for r in records],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    dup_report = [{"sha256": h, "images": [r.global_ordinal for r in g]} for h, g in groups.items() if len(g) > 1]
    (meta / "exact_duplicates.json").write_text(json.dumps(dup_report, indent=2), encoding="utf-8")

    if args.contact_sheets:
        contact_sheets(records, args.output, cfg)

    (args.output / "summary.json").write_text(json.dumps({
        "collection_id": cfg["collection_id"],
        "source_docx_count": len(docs),
        "image_count": len(records),
        "first_image": records[0].global_ordinal if records else None,
        "last_image": records[-1].global_ordinal if records else None,
        "derived": args.derive,
        "contact_sheets": args.contact_sheets,
    }, indent=2), encoding="utf-8")

    preprocess_summary()
    logging.info("Done: %d images -> %s", len(records), args.output)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Preprocessing failed: %s", exc)
        raise SystemExit(1)
