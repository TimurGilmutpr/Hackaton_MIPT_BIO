# Chest CT — ZIP → LungMask → Pathology Excel

## Прошу предварительно разместить в папке /data предоставленные zip архивы: pneumotorax_anon.zip,  pneumonia_anon.zip, norma_anon.zip

## Что делает
Берёт ZIP с DICOM-исследованием грудной клетки, распаковывает, конвертирует срезы в PNG,  
применяет **OGK (lungmask + морфология) фильтр лёгких** и классифицирует «патология / нет».  
Результат сохраняется в `results.xlsx`.  

---

## Состав проекта
- **pipeline_infer_only_with_lung_mask.py** — основной скрипт инференса (ZIP → DICOM → PNG → фильтр → инференс → Excel).  
- **pipeline_infer_only.py** — упрощённый инференс без фильтрации.  
- **Prediction_tool.py** — загрузка чекпойнта, трансформы, инференс (`predict_single` / `predict_batch`).  
- **Classification_library.py** — код обучения моделей (TinyCNN, CustomNet, EfficientNet).  
- **making_data.ipynb** — подготовка данных.  
- **learning.ipynb** — обучение моделей.  
- **filtering_lung.ipynb** — эксперименты с фильтрацией лёгких.  
- **models/*.pth, label_encoder.pkl** — веса модели и кодировщик классов.  

---

## Установка
```bash
pip install torch torchvision pandas numpy pillow pydicom scikit-learn matplotlib
pip install opencv-python scikit-image scipy tqdm lungmask SimpleITK openpyxl
```

---

## Запуск (CLI)

```bash
python pipeline_infer_only_with_lung_mask.py \
    --data_dir ./data \
    --out_excel results.xlsx \
    --workdir ./workdir \
    --image_size 512 \
    --min_pixels 5000
```

или без фильтрации:

```bash
python pipeline_infer_only.py \
    --data_dir ./data \
    --out_excel results.xlsx \
    --workdir ./workdir \
    --image_size 512
```

---

## Запуск в Jupyter

```python
%pip install torch torchvision pandas numpy pillow pydicom scikit-learn matplotlib
%pip install opencv-python scikit-image scipy tqdm lungmask SimpleITK openpyxl

!python pipeline_infer_only_with_lung_mask.py --data_dir ./data --out_excel results.xlsx --workdir ./workdir --image_size 512

import pandas as pd
pd.read_excel("results.xlsx").head()
```

---

## Формат `results.xlsx`

* `path_to_study` — путь к PNG-срезу  
* `study_uid`, `series_uid` — идентификаторы DICOM  
* `probability_of_pathology` — вероятность патологии  
* `pathology` — 1/0 (0 = норма, 1 = патология)  
* `processing_status` — `Success` / `Failure`  
* `time_of_processing` — время обработки серии  
* `error` — сообщение об ошибке  

---

## Структура проекта

```
project/
├─ pipeline_infer_only_with_lung_mask.py
├─ pipeline_infer_only.py
├─ Prediction_tool.py
├─ Classification_library.py
├─ making_data.ipynb
├─ learning.ipynb
├─ filtering_lung.ipynb
├─ models/
│    ├─ best_model.pth
│    └─ label_encoder.pkl
├─ data/
│    ├─ study1.zip
│    ├─ study2.zip
│    └─ ...
└─ results.xlsx
```
