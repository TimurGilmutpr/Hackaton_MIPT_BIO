# ============================================
# Classification_library.py  (simplified/stable, grayscale 512)
# ============================================

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------------------
# Utils
# --------------------------------------------

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    """Keeps track of the most recent, average, sum, and count of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


# --------------------------------------------
# Losses
# --------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss для несбалансированных данных.
    alpha: Tensor[num_classes] или float/None
    gamma: степень фокусировки
    """
    def __init__(self, alpha: Optional[Union[torch.Tensor, float]] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (list, np.ndarray)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", None if alpha is None else alpha.float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C]; target: [B]
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# --------------------------------------------
# Schedulers
# --------------------------------------------

class WarmupCosine:
    """
    Тёплый старт + косинусное затухание.
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 5e-6):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs))
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = -1

    def step(self):
        self.last_epoch += 1
        for i, g in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.last_epoch < self.warmup_epochs:
                # линейный разогрев от 0 до base_lr
                lr = base_lr * float(self.last_epoch + 1) / max(1, self.warmup_epochs)
            else:
                # косинус от base_lr до min_lr
                e = self.last_epoch - self.warmup_epochs
                E = max(1, self.total_epochs - self.warmup_epochs)
                cos = 0.5 * (1 + math.cos(math.pi * e / E))
                lr = self.min_lr + (base_lr - self.min_lr) * cos
            g["lr"] = lr


# --------------------------------------------
# Models
# --------------------------------------------

class TinyCNN(nn.Module):
    """
    Очень простой и устойчивый к переобучению бэкбон для 1-канальных 512x512.
    ~1.2M параметров. Подходит как базовый, быстрый и «тихий».
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 1, dropout: float = 0.2):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # /2
            )

        self.features = nn.Sequential(
            block(in_channels, 32),   # 512 -> 256
            block(32, 64),            # 256 -> 128
            block(64, 128),           # 128 -> 64
            block(128, 256),          # 64  -> 32
            block(256, 256),          # 32  -> 16
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


class CustomNet(nn.Module):
    """
    Лёгкий резервный вариант, совместимый с Prediction_tool.
    Можно не использовать: TinyCNN — основной.
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 1, width: int = 32, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 512->256
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(width, width*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256->128
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(width*2, width*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(width*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128->64
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(width*4, width*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(width*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64->32
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width*4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.head(x)
        return x


# --------------------------------------------
# Datasets
# --------------------------------------------

class SimpleImageDataset(Dataset):
    """
    Универсальный датасет:
    - Если есть столбец `image` с numpy.ndarray — берём его.
    - Иначе читаем путь из `path` (по умолчанию) и грузим PIL Image.
    Обязательно наличие метки в `label_col`.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        path_col: str = "path",
        image_col: str = "image",
        transform: Optional[transforms.Compose] = None,
        in_channels: int = 1,
    ):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.path_col = path_col
        self.image_col = image_col
        self.transform = transform
        self.in_channels = in_channels

        if label_col not in self.df.columns:
            raise ValueError(f"'{label_col}' not found in dataframe columns: {self.df.columns.tolist()}")

    def __len__(self):
        return len(self.df)

    def _ensure_pil(self, arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            img = Image.fromarray(arr.astype(np.uint8))
        elif arr.ndim == 3:
            # возьмём первый канал или среднее
            if arr.shape[2] == 1:
                img = Image.fromarray(arr[:, :, 0].astype(np.uint8))
            else:
                # усредним по каналам и вернём «серое»
                gray = arr.mean(axis=2).astype(np.uint8)
                img = Image.fromarray(gray)
        else:
            raise ValueError("Unsupported array shape for image.")
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y = int(row[self.label_col])

        if self.image_col in self.df.columns and isinstance(row[self.image_col], np.ndarray):
            img = self._ensure_pil(row[self.image_col])
        else:
            p = row[self.path_col]
            if not isinstance(p, str):
                raise ValueError(f"Row {idx}: expected string path in '{self.path_col}', got {type(p)}")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image not found: {p}")
            img = Image.open(p)

        # приведение к grayscale
        if self.in_channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            # минимально безопасное приведение в тензор
            t = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.Grayscale(num_output_channels=self.in_channels),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*self.in_channels, [0.5]*self.in_channels),
            ])
            img = t(img)

        return img, y


# --------------------------------------------
# Trainer
# --------------------------------------------

@dataclass
class TrainConfig:
    num_classes: int
    in_channels: int = 1
    image_size: int = 512
    backbone: str = "tiny_cnn"   # 'tiny_cnn' | 'custom'
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 40
    early_stop_patience: int = 6
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None
    seed: int = 42
    device: str = "auto"
    # трансформы
    rotate_deg: int = 5  # лёгкая геометрия
    # если val_df=None и нужна валидация из train_df:
    val_split: float = 0.2  # стратифицировано


class Learning_rocks:
    """
    Упрощённый и надёжный тренер.
    Ожидает train_df/val_df с колонками:
      - label (int)
      - path (str)  ИЛИ  image (np.ndarray)
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        config: Optional[TrainConfig] = None,
        label_col: str = "label",
        path_col: str = "path",
        image_col: str = "image",
        class_names: Optional[List[str]] = None,  # можно сохранить имена в чекпоинт
    ):
        self.cfg = config or TrainConfig(num_classes=int(train_df[label_col].nunique()))
        seed_everything(self.cfg.seed)

        # device
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        print(f"[Device] {self.device}")

        self.label_col = label_col
        self.path_col = path_col
        self.image_col = image_col
        self.class_names = class_names

        # split если не дали валид
        if val_df is None and self.cfg.val_split > 0:
            self.train_df, self.val_df = self._stratified_split(train_df, self.cfg.val_split, label_col)
        else:
            self.train_df = train_df.reset_index(drop=True)
            self.val_df = None if val_df is None else val_df.reset_index(drop=True)

        self.num_classes = self.cfg.num_classes
        self.in_channels = self.cfg.in_channels

        # transforms & data
        self._build_transforms(self.cfg.image_size)
        self._build_datasets_and_loaders()

        # model
        self._build_model()

        # optimizer & scheduler & loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = WarmupCosine(
            self.optimizer,
            warmup_epochs=max(1, self.cfg.num_epochs // 15),
            total_epochs=self.cfg.num_epochs,
            min_lr=5e-6,
        )

        alpha = None
        if self.cfg.class_weights is not None:
            alpha = torch.tensor(self.cfg.class_weights, dtype=torch.float32, device=self.device)
        self.criterion = FocalLoss(alpha=alpha, gamma=self.cfg.focal_gamma, reduction="mean")

        # bookkeeping
        self.best_val_loss = float("inf")
        self.best_state: Optional[Dict[str, Union[str, int, dict]]] = None
        self.history = {"train_loss": [], "val_loss": []}

    # -------------- splits --------------

    @staticmethod
    def _stratified_split(df: pd.DataFrame, val_split: float, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
        idx = np.arange(len(df))
        (tr_idx, va_idx), = sss.split(idx, df[label_col].values)
        return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)

    # -------------- transforms --------------

    def _build_transforms(self, image_size: int):
        image_size = int(image_size)
        mean = [0.5] * self.in_channels
        std = [0.5] * self.in_channels
        to_ch = transforms.Grayscale(num_output_channels=self.in_channels)

        self.train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomRotation(self.cfg.rotate_deg, fill=0),
            to_ch,
            transforms.ToTensor(),                # [0,1]
            transforms.Normalize(mean=mean, std=std),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            to_ch,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # -------------- data --------------

    def _build_datasets_and_loaders(self):
        self.train_dataset = SimpleImageDataset(
            self.train_df,
            label_col=self.label_col,
            path_col=self.path_col,
            image_col=self.image_col,
            transform=self.train_transform,
            in_channels=self.in_channels,
        )

        self.val_dataset = None
        if self.val_df is not None and len(self.val_df) > 0:
            self.val_dataset = SimpleImageDataset(
                self.val_df,
                label_col=self.label_col,
                path_col=self.path_col,
                image_col=self.image_col,
                transform=self.val_transform,
                in_channels=self.in_channels,
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False,
        )

        self.val_loader = None
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=(self.device.type == 'cuda'),
                drop_last=False,
            )

    # -------------- model --------------

    def _build_model(self):
        if self.cfg.backbone == "tiny_cnn":
            self.model = TinyCNN(num_classes=self.num_classes, in_channels=self.in_channels, dropout=0.2)
            self.arch_name = "tiny_cnn"
        elif self.cfg.backbone == "custom":
            self.model = CustomNet(num_classes=self.num_classes, in_channels=self.in_channels)
            self.arch_name = "custom"
        else:
            # дефолт — tiny
            self.model = TinyCNN(num_classes=self.num_classes, in_channels=self.in_channels, dropout=0.2)
            self.arch_name = "tiny_cnn"

        self.model = self.model.to(self.device)

    # -------------- train / validate --------------

    def _one_epoch(self, loader: DataLoader, train: bool = True) -> float:
        meter = AverageMeter()
        if train:
            self.model.train()
        else:
            self.model.eval()

        for batch in loader:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # современный AMP (убирает FutureWarning)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            meter.update(loss.item(), x.size(0))

        return meter.avg

    def learn_model(self, verbose: bool = True):
        patience = self.cfg.early_stop_patience
        patience_ctr = 0

        for epoch in range(self.cfg.num_epochs):
            train_loss = self._one_epoch(self.train_loader, train=True)

            val_loss = train_loss
            if self.val_loader is not None:
                val_loss = self._one_epoch(self.val_loader, train=False)

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if verbose:
                msg = f"Epoch {epoch+1}/{self.cfg.num_epochs} | TrainLoss: {train_loss:.4f}"
                if self.val_loader is not None:
                    msg += f" | ValLoss: {val_loss:.4f}"
                print(msg)

            # --- ранний стоп по val_loss ---
            metric = val_loss if self.val_loader is not None else train_loss
            if metric < self.best_val_loss - 1e-6:
                self.best_val_loss = metric
                self.best_state = {
                    "arch": self.arch_name,
                    "state_dict": self.model.state_dict(),
                    "num_classes": self.num_classes,
                    "in_channels": self.in_channels,
                    "image_size": self.cfg.image_size,
                    "class_names": self.class_names,
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    if verbose:
                        print("Early stopping.")
                    break

        # в конце — загрузим лучшее
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state["state_dict"])

    # -------------- helpers --------------

    def print_split_sizes(self):
        """Безопасно печатает размеры трена/валида."""
        tr = len(self.train_df) if isinstance(self.train_df, pd.DataFrame) else 0
        va = len(self.val_df) if self.val_df is not None and isinstance(self.val_df, pd.DataFrame) else 0
        print(f"Train/Val sizes: {tr} / {va}")

    # -------------- save / export --------------

    def save_checkpoint(self, filepath: str):
        if self.best_state is None:
            # если по какой-то причине не сохранили — сохраним текущую
            state = {
                "arch": self.arch_name,
                "state_dict": self.model.state_dict(),
                "num_classes": self.num_classes,
                "in_channels": self.in_channels,
                "image_size": self.cfg.image_size,
                "class_names": self.class_names,
            }
        else:
            state = self.best_state

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(state, filepath)
        print(f"[Checkpoint] saved to: {filepath}")

    # -------------- inference helper --------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Быстрый инференс для батча путей/массива изображений (как в датасете).
        Возвращает softmax вероятности.
        """
        ds = SimpleImageDataset(
            df,
            label_col=self.label_col,
            path_col=self.path_col,
            image_col=self.image_col,
            transform=self.val_transform,
            in_channels=self.in_channels,
        )
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=False,
        )

        self.model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(self.device, non_blocking=True)
                logits = self.model(x)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                preds.append(p)
        return np.concatenate(preds, axis=0)

    # -------------- evaluation & plots --------------

    def evaluate_df(self, df: pd.DataFrame) -> Dict[str, Union[float, np.ndarray]]:
        """
        Считает accuracy, macro-F1 и confusion matrix на любом DataFrame с label.
        """
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        probs = self.predict_proba(df)
        y_true = df[self.label_col].to_numpy()
        y_pred = probs.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average='macro')
        cm  = confusion_matrix(y_true, y_pred)
        return {"accuracy": float(acc), "f1_macro": float(f1m), "confusion_matrix": cm}

    def plot_history(self, show: bool = True, save_path: Optional[str] = None):
        """
        Строит графики train/val loss по эпохам.
        """
        import matplotlib.pyplot as plt
        train = self.history.get("train_loss", [])
        val = self.history.get("val_loss", [])

        if len(train) == 0:
            print("No history to plot yet.")
            return

        plt.figure(figsize=(7, 5))
        plt.plot(range(1, len(train)+1), train, label="Train loss")
        if len(val) == len(train):
            plt.plot(range(1, len(val)+1), val, label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training history")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"[Plot] saved to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()


# Основной блок с исправлением
if __name__ == "__main__":
    # Пример использования с исправлением
    train_df = pd.DataFrame()  # ваши тренировочные данные
    val_df = pd.DataFrame()    # валидационные данные (может быть None)
    
    config = TrainConfig(num_classes=2)
    trainer = Learning_rocks(train_df, val_df, config=config)
    
    trainer.learn_model()
    trainer.save_checkpoint("models/best_model.pth")
    
    # Исправленная строка - используем метод класса вместо прямого доступа
    trainer.print_split_sizes()