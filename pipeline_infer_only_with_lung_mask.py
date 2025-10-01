
"""
Только инференс: DICOM ZIP → PNG → фильтр ОГК (грудная клетка) → инференс → results.xlsx

Обучение здесь НЕ выполняется. Скрипт ожидает наличие:
  - models/best_model.pth
  - models/label_encoder.pkl

Логика фильтра ОГК бережно встроена:
  - попытка сегментации легких через lungmask с тремя окнами
  - при неудаче — морфологический фолбэк
  - в инференс попадают только срезы, где площадь маски ≥ min_pixels

Запуск:
  python pipeline_infer_only.py \
    --data_dir ./data \
    --out_excel results.xlsx \
    --workdir ./workdir \
    --image_size 512 \
    --min_pixels 5000
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

# DICOM
try:
    import pydicom
except Exception as e:
    print("Please install pydicom: pip install pydicom", file=sys.stderr)
    raise

# Imaging / filters (as in user's code)
import re
import cv2
import SimpleITK as sitk
from tqdm import tqdm

# lungmask + skimage + scipy 
from lungmask import mask as lungmask
import skimage.morphology as morph
import skimage.measure as meas
from scipy.ndimage import binary_fill_holes

# Локальный предиктор
from Prediction_tool import Predict  # must be importable


# -----------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_dicom(path: Path) -> bool:
    if path.suffix.lower() == ".dcm":
        return True
    try:
        with path.open("rb") as f:
            preamble = f.read(132)
        return preamble[-4:] == b"DICM"
    except Exception:
        return False


def normalize_img(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - mn) / (mx - mn) * 255.0
    return out.clip(0, 255).astype(np.uint8)


def dicom_to_pngs(dcm_path: Path, out_dir: Path) -> List[Path]:
    pngs: List[Path] = []
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True)
        px = ds.pixel_array
        if px.ndim == 2:
            arr = normalize_img(px)
            img = Image.fromarray(arr)
            out = out_dir / (dcm_path.stem + ".png")
            img.save(out)
            pngs.append(out)
        elif px.ndim == 3:
            for i in range(px.shape[0]):
                arr = normalize_img(px[i])
                img = Image.fromarray(arr)
                out = out_dir / f"{dcm_path.stem}_frame_{i+1:03d}.png"
                img.save(out)
                pngs.append(out)
    except Exception:
        pass
    return pngs


def extract_uids(ds) -> Tuple[str, str]:
    study_uid = getattr(ds, "StudyInstanceUID", "") or ""
    series_uid = getattr(ds, "SeriesInstanceUID", "") or ""
    return str(study_uid), str(series_uid)



# -----------------------------
# ПОДГОТОВКА ДАННЫй
# -----------------------------
def prepare_series(data_dir: Path, workdir: Path) -> Tuple[pd.DataFrame, Dict[str, List[Path]]]:
    """
    Распаковывает архивы, собирает DICOM-файлы и конвертирует в PNG.
    Возвращает:
      df_meta — строки на каждый DICOM
      series_to_pngs — словарь "<study>|<series>" -> список PNG путей

    """
    safe_mkdir(workdir)
    images_dir = workdir / "images"
    safe_mkdir(images_dir)

    rows = []
    series_to_pngs: Dict[str, List[Path]] = {}

    zips = sorted(list(data_dir.glob("*.zip")))
    if not zips:
        print(f"[WARN] No ZIP archives in {data_dir}", file=sys.stderr)

    for z in zips:
        out_dir = workdir / z.stem
        safe_mkdir(out_dir)
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(out_dir)
        except Exception as e:
            rows.append(dict(
                zip_name=z.name, dcm_path="", png_paths=[],
                study_uid="", series_uid="",
                representative_png="", error=f"Unzip error: {repr(e)}"
            ))
            continue

        dcm_files: List[Path] = []
        for p in out_dir.rglob("*"):
            if p.is_file() and (p.suffix.lower() in (".dcm", "") or is_dicom(p)):
                dcm_files.append(p)

        if not dcm_files:
            rows.append(dict(
                zip_name=z.name, dcm_path="", png_paths=[],
                study_uid="", series_uid="",
                representative_png="", error="No DICOMs found"
            ))
            continue

        for dcm in dcm_files:
            try:
                ds = pydicom.dcmread(str(dcm), force=True)
                study_uid, series_uid = extract_uids(ds)
                out_png_dir = images_dir / z.stem
                safe_mkdir(out_png_dir)
                pngs = dicom_to_pngs(dcm, out_png_dir)
                rep = pngs[len(pngs)//2] if pngs else ""
                key = f"{study_uid}|{series_uid}"
                series_to_pngs.setdefault(key, []).extend(pngs)

                rows.append(dict(
                    zip_name=z.name, dcm_path=str(dcm), png_paths=[str(p) for p in pngs],
                    study_uid=study_uid, series_uid=series_uid,
                    representative_png=str(rep), error=""
                ))
            except Exception as e:
                rows.append(dict(
                    zip_name=z.name, dcm_path=str(dcm), png_paths=[],
                    study_uid="", series_uid="",
                    representative_png="", error=f"DICOM parse/convert error: {repr(e)}"
                ))

    df_meta = pd.DataFrame(rows)
    return df_meta, series_to_pngs


# -----------------------------
# ФИЛЬТР ОГК
# -----------------------------
num_re = re.compile(r'(\d+)')

def sort_key(fname: str):
    m = num_re.findall(fname)
    return (int(m[-1]) if m else 0, fname)

def build_volume_from_files(png_files: List[Path]) -> Tuple[np.ndarray, List[Path]]:
    files_sorted = sorted(png_files, key=lambda p: sort_key(p.name))
    if not files_sorted:
        return np.zeros((0,512,512), dtype=np.uint8), []
    slices = []
    kept = []
    for p in files_sorted:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (512,512):
            img = cv2.resize(img, (512,512), interpolation=cv2.INTER_NEAREST)
        slices.append(img)
        kept.append(p)
    if not slices:
        return np.zeros((0,512,512), dtype=np.uint8), []
    vol_u8 = np.stack(slices, axis=0)  # (z,y,x)
    return vol_u8, kept

def u8_to_hu_approx(vol_u8: np.ndarray, W: float, L: float) -> np.ndarray:
    return (vol_u8.astype(np.float32) / 255.0) * W + (L - W/2.0)

def sitk_from_hu(vol_hu: np.ndarray) -> sitk.Image:
    vol = vol_hu.astype(np.int16, copy=False)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img

def run_lungmask_best(vol_u8: np.ndarray) -> np.ndarray:
    windows = [(1500, -600), (1600, -600), (350, 40)]
    best_mask = None
    best_sum = -1
    for W, L in windows:
        hu = u8_to_hu_approx(vol_u8, W, L)
        img = sitk_from_hu(hu)
        try:
            m = lungmask.apply(img)  # (z,y,x), 0/1
            s = m.sum()
            if s > best_sum:
                best_sum = s
                best_mask = m
        except Exception:
            continue
    if best_mask is None:
        return np.zeros_like(vol_u8, dtype=np.uint8)
    return best_mask.astype(np.uint8)

def fallback_chest_mask(vol_u8: np.ndarray) -> np.ndarray:
    Z, H, W = vol_u8.shape
    out = np.zeros_like(vol_u8, dtype=np.uint8)
    for i in range(Z):
        sl = vol_u8[i]
        # Бинаризация тела
        _, thr_img = cv2.threshold(sl, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        thr_val, _ = cv2.threshold(sl, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        body = (sl >= thr_val).astype(np.uint8)

        # Воздух/лёгкие
        thr_lung = int(max(0, thr_val * 0.6))
        lungs = (sl <= thr_lung).astype(np.uint8)
        lungs = cv2.bitwise_and(lungs, body)

        lungs = morph.binary_opening(lungs, morph.disk(3))
        lungs = morph.binary_closing(lungs, morph.disk(5))
        lungs = morph.remove_small_objects(lungs.astype(bool), 500)

        lab = meas.label(lungs, connectivity=2)
        props = sorted(meas.regionprops(lab), key=lambda p: p.area, reverse=True)
        keep = [p.label for p in props[:3]]
        if keep:
            m = np.isin(lab, keep)
            m = morph.binary_closing(m, morph.disk(7))
            m = binary_fill_holes(m)
            out[i] = m.astype(np.uint8)
    return out

def filter_series_pngs(series_to_pngs: Dict[str, List[Path]], min_pixels: int) -> Dict[str, List[Path]]:
    """
    Применяет фильтр ОГК к PNG-файлам серии.
    Возвращает отфильтрованный словарь "<study>|<series>" -> список PNG путей
    """
    filtered: Dict[str, List[Path]] = {}
    for key, pngs in tqdm(series_to_pngs.items(), desc="OGK filter per series"):
        vol_u8, files = build_volume_from_files(pngs)
        if vol_u8.size == 0:
            filtered[key] = pngs  # nothing to do
            continue
        # 1) lungmask with three windows
        mask = run_lungmask_best(vol_u8)
        # 2) fallback if mask too small
        if mask.sum() < min_pixels:
            mask_fb = fallback_chest_mask(vol_u8)
            if mask_fb.sum() > mask.sum():
                mask = mask_fb
        # 3) keep slices by area
        kept = []
        for i, p in enumerate(files):
            if i < mask.shape[0] and mask[i].sum() >= min_pixels:
                kept.append(p)
        # if nothing passed, keep original to avoid dropping the series entirely
        filtered[key] = kept if kept else pngs
    return filtered


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_excel", type=str, default="results.xlsx")
    ap.add_argument("--workdir", type=str, default="workdir")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--min_pixels", type=int, default=5000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    workdir = Path(args.workdir)
    safe_mkdir(workdir)

    # 1)   Подготовить серии: распаковка, DICOM → PNG
    print("[1/3] Preparing series ...")
    df_meta, series_to_pngs = prepare_series(data_dir, workdir)
    (workdir / "df_meta.csv").write_text(df_meta.to_csv(index=False))

    # 2) Применить фильтр ОГК
    print("[2/3] Applying OGK (chest) filter ...")
    # cast strings to Path for consistency
    series_to_pngs_path: Dict[str, List[Path]] = {k: [Path(p) if not isinstance(p, Path) else p for p in v]
                                                  for k, v in series_to_pngs.items()}
    filtered_series_to_pngs = filter_series_pngs(series_to_pngs_path, args.min_pixels)

    # 3)  Инференс
    print("[3/3] Inference ...")
    model_path = Path("models/best_model.pth")
    le_path = Path("models/label_encoder.pkl")

    rows = []
    if not (model_path.exists() and le_path.exists()):
        meta_groups = df_meta.groupby(['study_uid', 'series_uid'], dropna=False)
        for (study_uid, series_uid), grp in meta_groups:
            rows.append(dict(
                path_to_study="", study_uid=str(study_uid or ""), series_uid=str(series_uid or ""),
                probability_of_pathology=0.0, pathology=-1,
                processing_status="Failure", time_of_processing=0.0,
                error="Model or label encoder not found: expected models/best_model.pth and models/label_encoder.pkl"
            ))
        pd.DataFrame(rows).to_excel(args.out_excel, index=False)
        print(f"[DONE] Report (failures) → {args.out_excel}")
        return

    # 
    import pickle
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    class_names = list(le.classes_)

    normal_aliases = {"norma", "normal", "no_finding", "healthy"}
    normal_name = None
    for n in class_names:
        if n.lower() in normal_aliases or "norma" in n.lower():
            normal_name = n
            break
    if normal_name is None and class_names:
        normal_name = class_names[0]

    predictor = Predict(num_classes=len(class_names),
                        device='auto',
                        model_path=str(model_path),
                        class_names=class_names)

    meta_groups = df_meta.groupby(['study_uid', 'series_uid'], dropna=False)

    for (study_uid, series_uid), grp in meta_groups:
        t0 = time.time()
        status = "Success"
        err = ""
        prob_path = 0.0
        patho = -1
        path_to_study = ""
        try:
            key = f"{study_uid}|{series_uid}"
            pngs = filtered_series_to_pngs.get(key, [])
            if not pngs:
                # фаллбэк: representative из метаданных
                reps = [p for p in grp['representative_png'].tolist() if p]
                pngs = [Path(reps[len(reps)//2])] if reps else []
            if not pngs:
                raise RuntimeError("No PNG frames for series after filtering")

            # представительный срез
            rep = str(pngs[len(pngs)//2])
            path_to_study = rep

            out = predictor.predict_batch([rep])[0]
            probs = np.asarray(out['probabilities'], dtype=float)

            if normal_name in class_names:
                p_norm = float(probs[class_names.index(normal_name)])
            else:
                p_norm = float(probs.max())
            prob_path = float(np.clip(1.0 - p_norm, 0.0, 1.0))

            pred_idx = int(np.argmax(probs))
            pred_name = class_names[pred_idx]
            patho = 0 if pred_name == normal_name else 1

        except Exception as e:
            status = "Failure"
            err = repr(e)

        dt = time.time() - t0
        rows.append(dict(
            path_to_study=path_to_study,
            study_uid=str(study_uid or ""),
            series_uid=str(series_uid or ""),
            probability_of_pathology=prob_path,
            pathology=patho,
            processing_status=status,
            time_of_processing=round(float(dt), 3),
            error=err
        ))

    df_out = pd.DataFrame(rows)[[
        "path_to_study",
        "study_uid",
        "series_uid",
        "probability_of_pathology",
        "pathology",
        "processing_status",
        "time_of_processing",
        "error"
    ]]
    df_out.to_excel(args.out_excel, index=False)
    print(f"[DONE] Report → {args.out_excel}")


if __name__ == "__main__":
    main()
