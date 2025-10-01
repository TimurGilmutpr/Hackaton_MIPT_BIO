
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference-only pipeline: DICOM ZIP → PNG → inference → results.xlsx

No training is performed here. The script expects:
  - models/best_model.pth
  - models/label_encoder.pkl

Usage:
  python pipeline_infer_only.py \
    --data_dir ./data \
    --out_excel results.xlsx \
    --workdir ./workdir \
    --image_size 512

Outputs:
  - Excel with required columns:
    path_to_study, study_uid, series_uid, probability_of_pathology,
    pathology, processing_status, time_of_processing, error
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

try:
    import pydicom
except Exception as e:
    print("Please install pydicom: pip install pydicom", file=sys.stderr)
    raise

from Prediction_tool import Predict  # must be importable


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


def prepare_series(data_dir: Path, workdir: Path) -> Tuple[pd.DataFrame, Dict[str, List[Path]]]:
    """
    Unzip archives, collect DICOMs, convert to PNGs.
    Returns:
      df_meta: rows per dicom file
      series_to_pngs: "<study>|<series>" -> list[PNG paths]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_excel", type=str, default="results.xlsx")
    ap.add_argument("--workdir", type=str, default="workdir")
    ap.add_argument("--image_size", type=int, default=512)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    workdir = Path(args.workdir)
    safe_mkdir(workdir)

    # 1) Prepare series
    print("[1/2] Preparing series ...")
    df_meta, series_to_pngs = prepare_series(data_dir, workdir)
    (workdir / "df_meta.csv").write_text(df_meta.to_csv(index=False))

    # 2) Inference using existing model
    print("[2/2] Inference ...")
    from pathlib import Path as _P
    model_path = _P("models/best_model.pth")
    le_path = _P("models/label_encoder.pkl")

    rows = []
    if not (model_path.exists() and le_path.exists()):
        # If model/label encoder missing, emit Failure for each series
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

    # Load classes
    import pickle
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    class_names = list(le.classes_)

    # Define normal class name
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
            pngs = series_to_pngs.get(key, [])
            if not pngs:
                # try representative_png from meta
                reps = [p for p in grp['representative_png'].tolist() if p]
                pngs = [Path(reps[len(reps)//2])] if reps else []
            if not pngs:
                raise RuntimeError("No PNG frames for series")

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
