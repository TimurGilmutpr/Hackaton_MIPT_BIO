# ============================================
# Prediction_tool.py  (compatible with simplified Classification_library)
# ============================================

from Classification_library import TinyCNN, CustomNet, torch, nn, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np

# ---------- helpers ----------

def _build_efficientnet_b0(num_classes: int, in_channels: int = 3, pretrained: bool = False):
    """
    Сборка torchvision EfficientNet-B0 с опцией 1-канального входа.
    Применяется только для обратной совместимости со старыми весами.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    # подменим первый conv при 1-канальном входе
    if in_channels == 1:
        conv = model.features[0][0]  # Conv2d(in=3, out=32, ...)
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                # усредняем RGB по каналам -> 1 канал
                new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True)
        model.features[0][0] = new_conv
    # финальный классификатор
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)
    return model


def _to_pil(img, force_mode: str):
    """
    Привести вход (path/PIL/ndarray/tensor) к PIL и конвертировать по force_mode: 'L' или 'RGB'.
    """
    if isinstance(img, str):
        pil = Image.open(img)
    elif isinstance(img, Image.Image):
        pil = img
    elif isinstance(img, np.ndarray):
        pil = Image.fromarray(img)
    elif isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 2:      # [H,W]
            pil = transforms.ToPILImage()(t)
        elif t.ndim == 3:    # [C,H,W] или [H,W,C]
            if t.shape[0] in (1, 3):    # [C,H,W]
                pil = transforms.ToPILImage()(t)
            else:                        # [H,W,C]
                pil = transforms.ToPILImage()(t.permute(2, 0, 1))
        else:
            raise ValueError(f"Unexpected tensor shape {tuple(t.shape)}")
    else:
        raise ValueError("Unsupported image type. Provide path, PIL.Image, numpy array, or torch.Tensor")

    return pil.convert(force_mode)


# ---------- main predictor ----------

class Predict:
    def __init__(self, num_classes=None, device='auto', model_path='best_model.pth', class_names=None):
        """
        num_classes: если None — берётся из чекпоинта; иначе проверяется на совпадение.
        device: 'auto' | 'cuda' | 'cpu'
        model_path: путь к torch.save({...})
        class_names: переопределяет имена классов; иначе берутся из чекпоинта (если сохранены)
        """
        # --- устройство ---
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif str(device).lower() in ['gpu', 'cuda']:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            try:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass

        self.model_path = model_path

        # --- загрузка чекпоинта ---
        # weights_only=False — чтобы гарантированно получить метаданные
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # поддержка разных ключей
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            # может быть чистый state_dict (как в новой библиотеке)
            state_dict = ckpt if isinstance(ckpt, dict) else ckpt

        # --- архитектура ---
        if isinstance(ckpt, dict) and 'arch' in ckpt:
            arch = ckpt['arch']
        else:
            # эвристики по ключам
            ks = list(state_dict.keys())
            if any(k.startswith('features.0.0') for k in ks):              # EfficientNet
                arch = 'effnet_b0'
            elif any(k.startswith('stem.0') for k in ks) and any('head' in k for k in ks):  # CustomNet
                arch = 'custom'
            elif any(k.startswith('features.0.0') is False for k in ks) and any(k.startswith('head.') for k in ks):
                arch = 'tiny_cnn'  # простая TinyCNN: есть head.*, нет features.0.0
            else:
                arch = 'custom'  # запасной вариант

        # --- число классов ---
        def _infer_num_classes_by_arch(sd: dict, arch_name: str) -> int:
            if arch_name == 'effnet_b0':
                if 'classifier.1.weight' in sd:
                    return sd['classifier.1.weight'].shape[0]
            elif arch_name == 'custom':
                # финальный слой в CustomNet: head.3.weight (Linear)
                if 'head.3.weight' in sd:
                    return sd['head.3.weight'].shape[0]
                # старые варианты:
                if 'classifier.5.weight' in sd:
                    return sd['classifier.5.weight'].shape[0]
            else:  # tiny_cnn
                if 'head.6.weight' in sd:
                    return sd['head.6.weight'].shape[0]
            # резерв: минимальный out_features среди всех 2D-весов
            candidates = [v for k, v in sd.items() if k.endswith('.weight') and isinstance(v, torch.Tensor) and v.ndim == 2]
            if not candidates:
                raise RuntimeError("Не удалось определить num_classes из state_dict")
            return min(t.shape[0] for t in candidates)

        inferred_num_classes = _infer_num_classes_by_arch(state_dict, arch)
        if num_classes is None:
            num_classes = inferred_num_classes
        elif num_classes != inferred_num_classes:
            print(f"[WARN] num_classes({num_classes}) != checkpoint({inferred_num_classes}). Использую {inferred_num_classes}.")
            num_classes = inferred_num_classes
        self.num_classes = int(num_classes)

        # --- входные каналы ---
        if isinstance(ckpt, dict) and 'in_channels' in ckpt:
            in_channels = int(ckpt['in_channels'])
        else:
            # попытаемся понять по первому conv
            if arch == 'effnet_b0':
                w0 = state_dict.get('features.0.0.weight', None)
            elif arch == 'custom':
                w0 = state_dict.get('stem.0.weight', None)
            else:  # tiny_cnn
                # первый conv: features.0.0.weight
                w0 = state_dict.get('features.0.0.weight', None)
            in_channels = int(w0.shape[1]) if (isinstance(w0, torch.Tensor) and w0.ndim == 4) else 1
        self.in_channels = in_channels

        # --- размер входа ---
        if isinstance(ckpt, dict) and 'image_size' in ckpt:
            self.image_size = int(ckpt['image_size'])
        else:
            # по новой библиотеке дефолт 512
            self.image_size = 512

        # --- имена классов ---
        saved_class_names = ckpt.get('class_names', None) if isinstance(ckpt, dict) else None
        self.class_names = class_names if class_names is not None else saved_class_names

        # --- сборка модели и загрузка весов ---
        if arch == 'effnet_b0':
            # старые веса могли быть с effnet — оставим поддержку
            self.model = _build_efficientnet_b0(
                num_classes=self.num_classes,
                in_channels=self.in_channels,
                pretrained=False
            )
        elif arch == 'tiny_cnn':
            self.model = TinyCNN(num_classes=self.num_classes, in_channels=self.in_channels, dropout=0.2)
        else:
            self.model = CustomNet(num_classes=self.num_classes, in_channels=self.in_channels)

        # пробуем строго, затем — мягко с предупреждением
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"[WARN] strict load failed: {e}\nTrying strict=False…")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[WARN] missing_keys={missing}, unexpected_keys={unexpected}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # --- тестовые трансформы ---
        if arch == 'effnet_b0' and self.in_channels == 3:
            eff_w = EfficientNet_B0_Weights.IMAGENET1K_V1
            mean, std = eff_w.meta['mean'], eff_w.meta['std']
            force_mode = 'RGB'
        else:
            # по умолчанию — как в тренинге: grayscale 512 с mean=std=0.5 (или RGB ImageNet, если 3 канала)
            if self.in_channels == 1:
                mean, std = [0.5], [0.5]
                force_mode = 'L'
            else:
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                force_mode = 'RGB'

        self.force_mode = force_mode
        self.test_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.Lambda(lambda img: img.convert(self.force_mode)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        print(f"[DEBUG] arch={arch}, in_channels={self.in_channels}, num_classes={self.num_classes}, image_size={self.image_size}, mode={self.force_mode}")

    # ----------------- ИНФЕРЕНС -----------------
    @torch.no_grad()
    def predict_single(self, img, tta: bool = False):
        """
        Возвращает dict: {'predicted_class', 'confidence', 'probabilities', 'label'(если есть)}
        img: path | PIL.Image | np.ndarray | torch.Tensor
        """
        pil = _to_pil(img, self.force_mode)
        x = self.test_transform(pil).unsqueeze(0).to(self.device)

        if not tta:
            logits = self.model(x)
        else:
            # простая TTA: исход + горизонтальный флип
            logits = (self.model(x) + self.model(torch.flip(x, dims=[3]))) / 2

        probs = torch.softmax(logits, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
        result = {
            'predicted_class': int(pred.item()),
            'confidence': float(conf.item()),
            'probabilities': probs.detach().cpu().numpy().tolist()
        }
        if self.class_names and 0 <= result['predicted_class'] < len(self.class_names):
            result['label'] = self.class_names[result['predicted_class']]
        return result

    @torch.no_grad()
    def predict_batch(self, imgs, tta: bool = False):
        """
        imgs: iterable путей/картинок/массивов/tensor — вернёт список dict'ов как в predict_single
        """
        pil_list = [_to_pil(img, self.force_mode) for img in imgs]
        x = torch.stack([self.test_transform(p) for p in pil_list], dim=0).to(self.device)

        if not tta:
            logits = self.model(x)
        else:
            logits = (self.model(x) + self.model(torch.flip(x, dims=[3]))) / 2

        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        results = []
        for i in range(len(imgs)):
            item = {
                'predicted_class': int(preds[i].item()),
                'confidence': float(confs[i].item()),
                'probabilities': probs[i].detach().cpu().numpy().tolist()
            }
            if self.class_names and 0 <= item['predicted_class'] < len(self.class_names):
                item['label'] = self.class_names[item['predicted_class']]
            results.append(item)
        return results
