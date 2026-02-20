from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes, VOCSegmentation
from torchvision.transforms import ColorJitter
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

VOC_NUM_CLASSES = 21
CITYSCAPES_NUM_CLASSES = 19
IGNORE_INDEX = 255

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CITYSCAPES_ID_TO_TRAIN_ID = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for source_id, train_id in {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}.items():
    CITYSCAPES_ID_TO_TRAIN_ID[source_id] = train_id


@dataclass
class SegmentationDataConfig:
    data_root: str = "data"

    train_dataset: str = "voc"
    val_dataset: str = "voc"

    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    input_height: int = 512
    input_width: int = 512
    normalize: bool = True
    train_hflip_prob: float = 0.5

    # Corruption options: none, gaussian_blur, gaussian_noise, jpeg, color_jitter
    train_corruption: Optional[str] = None
    val_corruption: Optional[str] = None
    train_corruption_severity: int = 1
    val_corruption_severity: int = 1

    # Resolution shift is simulated by downsample->upsample with this scale.
    train_resolution_scale: float = 1.0
    val_resolution_scale: float = 1.0

    # VOC options
    download_voc: bool = True
    voc_year: str = "2012"
    voc_train_split: str = "train"
    voc_val_split: str = "val"

    # Cityscapes options (manual download required)
    cityscapes_mode: str = "fine"
    cityscapes_train_split: str = "train"
    cityscapes_val_split: str = "val"
    cityscapes_target_type: str = "semantic"
    cityscapes_use_train_ids: bool = True


class SegmentationPairTransform:
    def __init__(
        self,
        size: Tuple[int, int],
        train: bool,
        hflip_prob: float = 0.5,
    ) -> None:
        self.size = size
        self.train = train
        self.hflip_prob = hflip_prob

    def __call__(self, image: Image.Image, target: Image.Image) -> Tuple[Image.Image, Image.Image]:
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        target = TF.resize(target, self.size, interpolation=InterpolationMode.NEAREST)

        if self.train and self.hflip_prob > 0.0 and torch.rand(1).item() < self.hflip_prob:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target


class SegmentationDatasetWrapper(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        pair_transform: SegmentationPairTransform,
        mask_to_tensor_fn,
        normalize: bool,
        corruption: Optional[str],
        corruption_severity: int,
        resolution_scale: float,
    ) -> None:
        self.base_dataset = base_dataset
        self.pair_transform = pair_transform
        self.mask_to_tensor_fn = mask_to_tensor_fn
        self.normalize = normalize
        self.corruption = corruption
        self.corruption_severity = corruption_severity
        self.resolution_scale = resolution_scale

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[idx]
        image, target = self.pair_transform(image, target)

        image = _apply_resolution_shift(image, self.resolution_scale)
        image = _apply_corruption(image, self.corruption, self.corruption_severity)

        image_tensor = TF.to_tensor(image)
        if self.normalize:
            image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)

        target_tensor = self.mask_to_tensor_fn(target)
        return image_tensor, target_tensor


def build_segmentation_dataloaders(config: SegmentationDataConfig):
    train_base, train_mask_to_tensor, train_meta = _build_base_dataset(config, config.train_dataset, train=True)
    val_base, val_mask_to_tensor, val_meta = _build_base_dataset(config, config.val_dataset, train=False)

    size = (config.input_height, config.input_width)
    train_dataset = SegmentationDatasetWrapper(
        base_dataset=train_base,
        pair_transform=SegmentationPairTransform(size=size, train=True, hflip_prob=config.train_hflip_prob),
        mask_to_tensor_fn=train_mask_to_tensor,
        normalize=config.normalize,
        corruption=config.train_corruption,
        corruption_severity=config.train_corruption_severity,
        resolution_scale=config.train_resolution_scale,
    )
    val_dataset = SegmentationDatasetWrapper(
        base_dataset=val_base,
        pair_transform=SegmentationPairTransform(size=size, train=False, hflip_prob=0.0),
        mask_to_tensor_fn=val_mask_to_tensor,
        normalize=config.normalize,
        corruption=config.val_corruption,
        corruption_severity=config.val_corruption_severity,
        resolution_scale=config.val_resolution_scale,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_meta": train_meta,
        "val_meta": val_meta,
    }


def _build_base_dataset(config: SegmentationDataConfig, dataset_name: str, train: bool):
    name = dataset_name.lower()
    root = Path(config.data_root)

    if name == "voc":
        split = config.voc_train_split if train else config.voc_val_split
        base = VOCSegmentation(
            root=str(root),
            year=config.voc_year,
            image_set=split,
            download=config.download_voc,
        )
        return base, _voc_mask_to_tensor, {
            "dataset": "voc",
            "split": split,
            "num_classes": VOC_NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
        }

    if name == "cityscapes":
        split = config.cityscapes_train_split if train else config.cityscapes_val_split
        city_root = _resolve_cityscapes_root(root / "cityscapes")
        base = Cityscapes(
            root=str(city_root),
            split=split,
            mode=config.cityscapes_mode,
            target_type=config.cityscapes_target_type,
        )
        mask_to_tensor = (
            _cityscapes_mask_to_train_id_tensor
            if config.cityscapes_use_train_ids
            else _raw_mask_to_tensor
        )
        num_classes = CITYSCAPES_NUM_CLASSES if config.cityscapes_use_train_ids else 34
        return base, mask_to_tensor, {
            "dataset": "cityscapes",
            "split": split,
            "num_classes": num_classes,
            "ignore_index": IGNORE_INDEX,
        }

    raise ValueError(f"Unsupported segmentation dataset: '{dataset_name}'")


def _resolve_cityscapes_root(city_root: Path) -> Path:
    left = city_root / "leftImg8bit"
    gt = city_root / "gtFine"
    if left.exists() and gt.exists():
        return city_root

    alt_left = city_root / "leftImg8bit_trainvaltest" / "leftImg8bit"
    alt_gt = city_root / "gtFine_trainvaltest" / "gtFine"
    if alt_left.exists() and alt_gt.exists():
        # Accept both common extraction layouts.
        try:
            if left.is_symlink() and not left.exists():
                left.unlink()
            if gt.is_symlink() and not gt.exists():
                gt.unlink()
            if not left.exists():
                left.symlink_to(alt_left.resolve())
            if not gt.exists():
                gt.symlink_to(alt_gt.resolve())
        except Exception:
            pass
        if left.exists() and gt.exists():
            return city_root

    raise FileNotFoundError(
        f"Cityscapes not found at '{city_root}'. "
        "Expected either data/cityscapes/{leftImg8bit,gtFine} "
        "or data/cityscapes/{leftImg8bit_trainvaltest/leftImg8bit,gtFine_trainvaltest/gtFine}."
    )


def _raw_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    return torch.as_tensor(np.asarray(mask, dtype=np.uint8), dtype=torch.long)


def _voc_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    return torch.as_tensor(np.asarray(mask, dtype=np.uint8), dtype=torch.long)


def _cityscapes_mask_to_train_id_tensor(mask: Image.Image) -> torch.Tensor:
    raw_mask = np.asarray(mask, dtype=np.uint8)
    remapped = CITYSCAPES_ID_TO_TRAIN_ID[raw_mask]
    return torch.as_tensor(remapped, dtype=torch.long)


def _apply_resolution_shift(image: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return image
    if scale <= 0.0:
        raise ValueError(f"resolution scale must be > 0, got {scale}")

    h, w = image.height, image.width
    reduced_h = max(1, int(round(h * scale)))
    reduced_w = max(1, int(round(w * scale)))

    reduced = image.resize((reduced_w, reduced_h), resample=Image.BILINEAR)
    return reduced.resize((w, h), resample=Image.BILINEAR)


def _apply_corruption(image: Image.Image, name: Optional[str], severity: int) -> Image.Image:
    if name is None or name.lower() in {"none", ""}:
        return image

    sev = max(1, min(5, int(severity)))
    normalized = name.lower()

    if normalized == "gaussian_blur":
        radius = {1: 0.4, 2: 0.8, 3: 1.2, 4: 1.8, 5: 2.5}[sev]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    if normalized == "gaussian_noise":
        sigma = {1: 8.0, 2: 12.0, 3: 18.0, 4: 26.0, 5: 38.0}[sev]
        arr = np.asarray(image).astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
        corrupted = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(corrupted)

    if normalized == "jpeg":
        quality = {1: 80, 2: 65, 3: 50, 4: 35, 5: 20}[sev]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if normalized == "color_jitter":
        strength = {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25}[sev]
        jitter = ColorJitter(
            brightness=strength,
            contrast=strength,
            saturation=strength,
            hue=min(0.5, strength / 2.0),
        )
        return jitter(image)

    raise ValueError(
        "Unsupported corruption. Use one of: "
        "none, gaussian_blur, gaussian_noise, jpeg, color_jitter"
    )


if __name__ == "__main__":
    cfg = SegmentationDataConfig(
        train_dataset="voc",
        val_dataset="cityscapes",
        batch_size=2,
        num_workers=0,
        input_height=256,
        input_width=256,
        download_voc=True,
        val_corruption="gaussian_blur",
        val_corruption_severity=2,
    )
    artifacts = build_segmentation_dataloaders(cfg)
    train_images, train_masks = next(iter(artifacts["train_loader"]))
    val_images, val_masks = next(iter(artifacts["val_loader"]))
    print("train batch:", train_images.shape, train_masks.shape, artifacts["train_meta"])
    print("val batch:", val_images.shape, val_masks.shape, artifacts["val_meta"])
