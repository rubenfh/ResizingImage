#!/usr/bin/env python3
"""batch_cadrage_excel.py – v1.6 (multi-onglets + redimensionnement + fond homogène)

Nouveautés principales :
1. **Parcours automatique de tous les onglets**
   - Si l’argument `--sheet` vaut « all » (valeur par défaut), chaque onglet
     du classeur est traité successivement. Les éventuels onglets ne contenant
     pas les deux colonnes obligatoires sont ignorés avec un message d’avertissement.
2. **Redimensionnement final** (par défaut : 1520x1900 px) inchangé ;
3. **Homogénéisation du fond** inchangée ;
4. **Options CLI** mises à jour :
   - `--sheet all` (ou nom d’un onglet spécifique).

Les autres fonctionnalités (détection adaptative de l’objet, téléchargement
automatique, etc.) restent inchangées.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Utilitaires couleurs / fond
# ---------------------------------------------------------------------------

def estimate_background_color(arr: np.ndarray, frac: float = 0.05) -> Tuple[int, int, int]:
    h, w, _ = arr.shape
    border = max(1, int(min(h, w) * frac))
    samples = np.concatenate([
        arr[:border].reshape(-1, 3),
        arr[-border:].reshape(-1, 3),
        arr[:, :border].reshape(-1, 3),
        arr[:, -border:].reshape(-1, 3),
    ])
    r, g, b = np.median(samples, axis=0)
    return int(r), int(g), int(b)


def homogenize_background(img: Image.Image, bg_color: Tuple[int, int, int], *,
                          bright_thr: int = 240, sat_thr: int = 15) -> Image.Image:
    """Force tous les pixels *presque blancs* (ou très proches du fond) à la
    couleur de fond pour garantir la pleine homogénéité."""
    arr = np.array(img)
    brightness = arr.mean(axis=2)
    saturation = np.max(arr, axis=2) - np.min(arr, axis=2)

    # pixels très clairs ou très proches de bg_color
    near_bg = np.linalg.norm(arr.astype(np.int16) - np.array(bg_color), axis=2) < 10
    mask = ((brightness > bright_thr) & (saturation < sat_thr)) | near_bg
    arr[mask] = bg_color
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Détection de l’objet
# ---------------------------------------------------------------------------

def _longest_contiguous_groups(idxs: np.ndarray) -> List[np.ndarray]:
    if idxs.size == 0:
        return []
    splits = np.where(np.diff(idxs) != 1)[0] + 1
    return np.split(idxs, splits)


def _bbox_from_mask(mask: np.ndarray, min_fraction: float, min_group: int) -> Tuple[int, int, int, int] | None:
    min_px_row = max(1, int(mask.shape[1] * min_fraction))
    min_px_col = max(1, int(mask.shape[0] * min_fraction))

    rows_bool = mask.sum(axis=1) >= min_px_row
    cols_bool = mask.sum(axis=0) >= min_px_col

    rows_idx = np.where(rows_bool)[0]
    cols_idx = np.where(cols_bool)[0]
    if rows_idx.size == 0 or cols_idx.size == 0:
        return None

    row_groups = [g for g in _longest_contiguous_groups(rows_idx) if len(g) >= min_group]
    col_groups = [g for g in _longest_contiguous_groups(cols_idx) if len(g) >= min_group]
    if not row_groups or not col_groups:
        return None

    rows = max(row_groups, key=len)
    cols = max(col_groups, key=len)
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


# --- LISTES D’ESSAI AUTOMATIQUE -------------------------------------------
_MIN_FRACTIONS = [0.02, 0.01, 0.005, 0.002, 0.001]
_COLOR_DELTAS  = [25, 20, 15, 10, 5]
_THRESHOLDS_G  = [200, 210, 220]
_MIN_BBOX_AREA = 16  # px²


def get_object_bbox(
    image: Image.Image,
    *,
    min_group: int = 3,
) -> Tuple[int, int, int, int]:
    """BBox robuste (v1.5) – identique à la v1.4, mais gardée pour contexte."""
    arr = np.array(image)
    bg_r, bg_g, bg_b = estimate_background_color(arr)

    gray = Image.fromarray(arr).convert("L").filter(ImageFilter.GaussianBlur(1))
    g_arr = np.array(gray)
    diff = np.linalg.norm(arr.astype(np.int16) - np.array([bg_r, bg_g, bg_b]), axis=2)

    best_bbox: Tuple[int, int, int, int] | None = None
    best_area = 0

    def consider(mask: np.ndarray, mf: float):
        nonlocal best_bbox, best_area
        bbox = _bbox_from_mask(mask, mf, min_group)
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)
        if area >= _MIN_BBOX_AREA and area > best_area:
            best_bbox, best_area = bbox, area

    for mf in _MIN_FRACTIONS:
        for thr in _THRESHOLDS_G:
            consider(g_arr < thr, mf)
        for cd in _COLOR_DELTAS:
            consider(diff > cd, mf)

    if best_bbox is None:
        raise ValueError("Objet non détecté : paramètres auto épuisés.")
    return best_bbox


# ---------------------------------------------------------------------------
# Redimensionnement / Padding
# ---------------------------------------------------------------------------

def resize_and_pad(img: Image.Image, target_w: int, target_h: int, bg_color: Tuple[int, int, int]) -> Image.Image:
    """Redimensionne *proportionnellement* l’image pour qu’elle tienne dans la
    zone cible, puis la centre sur une toile de fond `bg_color`."""
    scale = min(target_w / img.width, target_h / img.height)
    new_w, new_h = int(round(img.width * scale)), int(round(img.height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas.paste(resized, (x_off, y_off))
    return canvas


# ---------------------------------------------------------------------------
# Marges de référence & fonction de recadrage
# ---------------------------------------------------------------------------

def compute_reference_margins(ref_path: os.PathLike | str, *, min_group: int = 3) -> Dict[str, int]:
    img = Image.open(ref_path).convert("RGB")
    x0, y0, x1, y1 = get_object_bbox(img, min_group=min_group)
    return {"left": x0, "top": y0, "right": img.width - x1, "bottom": img.height - y1}


def get_background_color(image: Image.Image) -> Tuple[int, int, int]:
    pixels = np.array(image).reshape(-1, 3)
    r, g, b = Counter(map(tuple, pixels)).most_common(1)[0][0]
    return int(r), int(g), int(b)


def standardize_cadrage(
    image_path: os.PathLike | str,
    output_path: os.PathLike | str,
    ref_margins: Dict[str, int],
    *,
    background_color: Tuple[int, int, int] | None = None,
    margin_scale: float = 0.9,
    target_size: Tuple[int, int] | None = (1520, 1900),
    min_group: int = 3,
) -> None:
    """Recadre *puis* redimensionne l’image selon `target_size`.

    - `target_size=None` : pas de redimensionnement.
    - `margin_scale` réduit/augmente proportionnellement les marges de
      référence.
    """
    img = Image.open(image_path).convert("RGB")
    if background_color is None:
        background_color = get_background_color(img)

    # 1. Détection + découpe de l’objet
    x0, y0, x1, y1 = get_object_bbox(img, min_group=min_group)
    obj = img.crop((x0, y0, x1, y1))
    obj_w, obj_h = x1 - x0, y1 - y0

   # 2. Reconstitution avec marges standardisées
    scaled = {k: max(1, int(round(v * margin_scale))) for k, v in ref_margins.items()}
    new_w = obj_w + scaled["left"] + scaled["right"]
    new_h = obj_h + scaled["top"] + scaled["bottom"]

    background_color = (250, 248, 246)
    canvas = Image.new("RGB", (new_w, new_h), background_color)
    canvas.paste(obj, (scaled["left"], scaled["top"]))

    # 3. Homogénéisation du fond
    canvas = homogenize_background(canvas, background_color)

    # 4. Redimensionnement final
    if target_size is not None:
        target_w, target_h = target_size
        canvas = resize_and_pad(canvas, target_w, target_h, background_color)

    # 5. Sauvegarde
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=100)
    print("✅", output_path)


# ---------------------------------------------------------------------------
# Téléchargement + Excel
# ---------------------------------------------------------------------------

def download_image(url: str, dest_dir: os.PathLike | str) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlsplit(url)
    filename = urllib.parse.unquote(Path(parsed.path).name) or "image.jpg"
    filename = filename.split("?")[0]
    dest = dest_dir / filename
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{dest.stem}_{counter}{dest.suffix}"
        counter += 1
    print(f"⬇️  {url} → {dest.name}…", end=" ")
    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp.raw, f)
    print("ok")
    return dest


def process_excel_sheet(
    excel_path: os.PathLike | str,
    sheet_name: str,
    output_root: os.PathLike | str,
    *,
    margin_scale: float = 0.9,
    target_size: Tuple[int, int] | None = (1520, 1900),
    min_group: int = 3,
) -> None:
    print(f"📖 {excel_path} | Onglet : '{sheet_name}'")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    ref_col, img_col = "Reference model url", "Image Src"
    if ref_col not in df.columns or img_col not in df.columns:
        raise KeyError(f"Colonnes manquantes : '{ref_col}' ou '{img_col}' dans '{sheet_name}'.")
    ref_url = df[ref_col].dropna().astype(str).iloc[0]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ref_path = download_image(ref_url, tmp_dir)
        ref_margins = compute_reference_margins(ref_path, min_group=min_group)
        print("📐 Marges de référence :", ref_margins)
        out_dir = Path(output_root) / sheet_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for url in df[img_col].dropna().astype(str):
            try:
                loc = download_image(url, tmp_dir)
                dest = out_dir / loc.name
                standardize_cadrage(
                    loc,
                    dest,
                    ref_margins,
                    margin_scale=margin_scale,
                    target_size=target_size,
                    min_group=min_group,
                )
            except Exception as e:
                print(f"⚠️ {url} : {e}")
    print("🎉 Terminé →", out_dir)


def process_excel_workbook(
    excel_path: os.PathLike | str,
    output_root: os.PathLike | str,
    *,
    margin_scale: float = 0.9,
    target_size: Tuple[int, int] | None = (1520, 1900),
    min_group: int = 3,
) -> None:
    """Traite **tous** les onglets trouvés dans le classeur Excel."""
    xls = pd.ExcelFile(excel_path)
    print(f"🗂️ Classeur détecté avec {len(xls.sheet_names)} onglet(s) : {xls.sheet_names}")
    for sheet in xls.sheet_names:
        try:
            process_excel_sheet(
                excel_path,
                sheet,
                output_root,
                margin_scale=margin_scale,
                target_size=target_size,
                min_group=min_group,
            )
        except KeyError as e:
            print(f"ℹ️Onglet ignoré ('{sheet}') : {e}")
    print("🥳 Traitement complet du classeur terminé.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    p = argparse.ArgumentParser(
        description="Recadre et redimensionne les images listées dans un fichier Excel Shopify (détection auto).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("excel", help="Fichier .xlsx à traiter")
    p.add_argument("--sheet", default="all", help="Nom de l'onglet ou 'all' pour traiter tout le classeur")
    p.add_argument("--output-dir", default="outputs", help="Dossier de sortie")
    p.add_argument("--margin-scale", type=float, default=0.9, help="Échelle des marges (0-1)")
    p.add_argument("--target-size", default="1520x1900", help="Tailles cibles LxH, ex : 1520x1900 ou 800x800. 'none' pour désactiver.")
    args = p.parse_args()

    if args.target_size.lower() == "none":
        target_size = None
    else:
        try:
            w, h = map(int, args.target_size.lower().split("x"))
            target_size = (w, h)
        except Exception as e:
            raise ValueError("Format --target-size invalide. Attendu : LxH, ex : 800x1000") from e

    if args.sheet.lower() == "all":
        process_excel_workbook(
            args.excel,
            args.output_dir,
            margin_scale=args.margin_scale,
            target_size=target_size,
        )
    else:
        process_excel_sheet(
            args.excel,
            args.sheet,
            args.output_dir,
            margin_scale=args.margin_scale,
            target_size=target_size,
        )


if __name__ == "__main__":
    cli()
