"""
RavvdessDatasetChallenging: Unpaired dataset with realistic distribution challenges.

Simulates 3 types of distribution mismatch between modalities:
  1. Marginal Mismatch  – each modality sees a DIFFERENT subset of samples per class
  2. Domain Shift        – audio augmented with noise / image with color jitter+blur
  3. Label Imbalance     – one modality has skewed class frequencies (long-tail)

All challenges are applied ONLY at training time; val/test stay clean & paired.
"""

import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
from collections import defaultdict
import random
from PIL import Image, ImageFilter
import math


class RavvdessDatasetChallenging(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to the CSV file (audio_name, image_name, label).
    audio_dir, image_dir : str
        Root directories for .npy audio features and .jpg images.
    mode : str
        'train' | 'val' | 'test'.  Challenges only apply to 'train'.

    --- Challenge knobs (only effective when mode='train') ---
    marginal_mismatch : bool
        If True, audio and image pools are disjoint subsets *within* each class.
        `marginal_ratio` controls the split (e.g. 0.5 → 50 % for audio, 50 % for image).
    marginal_ratio : float in (0, 1)
        Fraction of class samples assigned to the audio pool; the rest go to image.

    domain_shift : bool
        If True, apply extra "corruption" augmentations to simulate modality domain gap.
    domain_shift_level : float in [0, 1]
        Severity of domain‑shift augmentations (0 = mild, 1 = heavy).

    label_imbalance : bool
        If True, one modality observes a long‑tail version of the label distribution.
    imbalance_modality : str  ('audio' | 'image')
        Which modality gets the imbalanced view.
    imbalance_factor : float > 1
        Ratio between most‑frequent and least‑frequent class (exponential decay).
    """

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        image_dir: str,
        mode: str = "train",
        # --- marginal mismatch ---
        marginal_mismatch: bool = False,
        marginal_ratio: float = 0.5,
        # --- domain shift ---
        domain_shift: bool = False,
        domain_shift_level: float = 0.5,
        # --- label imbalance ---
        label_imbalance: bool = False,
        imbalance_modality: str = "audio",
        imbalance_factor: float = 10.0,
        # --- random seed for reproducibility ---
        seed: int = 42,
    ):
        import pandas as pd

        self.df = pd.read_csv(csv_path, header=None)
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        self.mode = mode

        # ---- basic fields ----
        self.audio_files = self.df.iloc[:, 0].values
        self.image_files = self.df.iloc[:, 1].values
        unique_classes = sorted(self.df.iloc[:, 2].unique())
        self.num_classes = len(unique_classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        self.labels = np.array([self.class_to_idx[l] for l in self.df.iloc[:, 2].values])

        # ---- per‑class index lists ----
        self.indices_per_class = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.indices_per_class[lbl].append(idx)

        # ============================================================
        #  Challenge 1: Marginal Mismatch
        #  Split each class into disjoint audio / image pools so that
        #  the two modalities never share the exact same sample.
        # ============================================================
        self.marginal_mismatch = marginal_mismatch and (mode == "train")
        self.audio_pool = defaultdict(list)   # class -> list of row indices
        self.image_pool = defaultdict(list)

        if self.marginal_mismatch:
            rng = random.Random(seed)
            for cls, idxs in self.indices_per_class.items():
                shuffled = idxs.copy()
                rng.shuffle(shuffled)
                split = max(1, int(len(shuffled) * marginal_ratio))
                self.audio_pool[cls] = shuffled[:split]
                self.image_pool[cls] = shuffled[split:]
                # safety: if one pool is empty, copy from the other
                if len(self.image_pool[cls]) == 0:
                    self.image_pool[cls] = self.audio_pool[cls].copy()
                if len(self.audio_pool[cls]) == 0:
                    self.audio_pool[cls] = self.image_pool[cls].copy()
        else:
            # default: both modalities share the full pool
            for cls, idxs in self.indices_per_class.items():
                self.audio_pool[cls] = idxs
                self.image_pool[cls] = idxs

        # ============================================================
        #  Challenge 2: Domain Shift
        #  Extra corruptions that widen the modality gap.
        # ============================================================
        self.domain_shift = domain_shift and (mode == "train")
        self.domain_shift_level = domain_shift_level

        # ============================================================
        #  Challenge 3: Label Imbalance
        #  Build a sampling‑weight vector so one modality sees a
        #  long‑tail distribution  (exponential decay across classes).
        # ============================================================
        self.label_imbalance = label_imbalance and (mode == "train")
        self.imbalance_modality = imbalance_modality

        if self.label_imbalance:
            # Compute per‑class sampling probabilities (exponential decay)
            # Class 0 → most frequent, class K-1 → least frequent
            rng_imb = random.Random(seed + 1)
            class_order = list(range(self.num_classes))
            rng_imb.shuffle(class_order)  # random head/tail assignment
            self.imbalance_probs = {}
            for rank, cls in enumerate(class_order):
                # prob ∝ (1/imbalance_factor)^(rank / (K-1))
                exp = rank / max(self.num_classes - 1, 1)
                self.imbalance_probs[cls] = (1.0 / imbalance_factor) ** exp

            # Pre‑build the imbalanced pool per class by sub‑sampling
            self._imbalanced_pool = defaultdict(list)
            rng_sub = random.Random(seed + 2)
            for cls in range(self.num_classes):
                src = self.audio_pool[cls] if imbalance_modality == "audio" else self.image_pool[cls]
                keep = max(1, int(len(src) * self.imbalance_probs[cls]))
                sampled = rng_sub.sample(src, min(keep, len(src)))
                self._imbalanced_pool[cls] = sampled
        else:
            self._imbalanced_pool = None

        # ============================================================
        #  Transforms
        # ============================================================
        self._build_transforms()

        # ---- logging ----
        if mode == "train":
            self._print_config()

    # ------------------------------------------------------------------
    #  Transforms
    # ------------------------------------------------------------------
    def _build_transforms(self):
        """Build audio and image transforms. Domain‑shift adds extra corruption."""
        # ---------- audio ----------
        base_aud = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[-10.59], std=[85.66]),
        ]
        if self.domain_shift:
            # Gaussian noise injection (controlled by level)
            self.audio_noise_std = 0.05 + 0.20 * self.domain_shift_level  # 0.05‑0.25
        self.aud_transform = transforms.Compose(base_aud)

        # ---------- image ----------
        if self.mode == "train":
            base_img = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
            if self.domain_shift:
                lvl = self.domain_shift_level
                base_img += [
                    transforms.ColorJitter(
                        brightness=0.2 + 0.4 * lvl,
                        contrast=0.2 + 0.4 * lvl,
                        saturation=0.2 + 0.3 * lvl,
                        hue=0.05 + 0.10 * lvl,
                    ),
                    transforms.RandomGrayscale(p=0.1 + 0.2 * lvl),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                        p=0.2 + 0.3 * lvl,
                    ),
                    transforms.RandomPerspective(
                        distortion_scale=0.1 + 0.2 * lvl, p=0.3
                    ),
                ]
            base_img += [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            if self.domain_shift:
                # Random erasing as final corruption
                base_img.append(
                    transforms.RandomErasing(
                        p=0.1 + 0.2 * self.domain_shift_level,
                        scale=(0.02, 0.15),
                    )
                )
            self.img_transform = transforms.Compose(base_img)
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    # ------------------------------------------------------------------
    #  Logging
    # ------------------------------------------------------------------
    def _print_config(self):
        print("=" * 60)
        print("  RavvdessDatasetChallenging — Train Configuration")
        print("=" * 60)
        print(f"  Total samples       : {len(self.df)}")
        print(f"  Num classes          : {self.num_classes}")
        print(f"  Marginal mismatch   : {self.marginal_mismatch}")
        if self.marginal_mismatch:
            for cls in sorted(self.audio_pool):
                print(f"    class {cls}: audio_pool={len(self.audio_pool[cls])}, "
                      f"image_pool={len(self.image_pool[cls])}")
        print(f"  Domain shift         : {self.domain_shift}"
              + (f" (level={self.domain_shift_level:.2f})" if self.domain_shift else ""))
        print(f"  Label imbalance      : {self.label_imbalance}")
        if self.label_imbalance:
            print(f"    modality={self.imbalance_modality}, factor={1.0/min(self.imbalance_probs.values()):.1f}x")
            for cls in sorted(self._imbalanced_pool):
                orig = len(self.audio_pool[cls]) if self.imbalance_modality == "audio" else len(self.image_pool[cls])
                print(f"    class {cls}: orig={orig} → imbalanced={len(self._imbalanced_pool[cls])}")
        print("=" * 60)

    # ------------------------------------------------------------------
    #  Core
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        target_label = self.labels[index]

        # ---- select audio index ----
        if self.mode == "train":
            if self.label_imbalance and self.imbalance_modality == "audio":
                pool = self._imbalanced_pool[target_label]
            else:
                pool = self.audio_pool[target_label]
            audio_idx = random.choice(pool)
        else:
            audio_idx = index

        # ---- select image index ----
        if self.mode == "train":
            if self.marginal_mismatch:
                # image always comes from image_pool (disjoint from audio_pool)
                if self.label_imbalance and self.imbalance_modality == "image":
                    pool = self._imbalanced_pool[target_label]
                else:
                    pool = self.image_pool[target_label]
                img_idx = random.choice(pool)
            else:
                if self.label_imbalance and self.imbalance_modality == "image":
                    pool = self._imbalanced_pool[target_label]
                    img_idx = random.choice(pool)
                else:
                    img_idx = index
        else:
            img_idx = index

        # ---- load audio ----
        aud_path = os.path.join(self.audio_dir, self.audio_files[audio_idx])
        audio_np = np.load(aud_path)
        if audio_np.ndim == 3 and audio_np.shape[2] == 3:
            audio_np = audio_np[:, :, ::-1].copy()

        # domain shift: additive Gaussian noise on audio spectrogram
        if self.domain_shift:
            noise = np.random.randn(*audio_np.shape).astype(audio_np.dtype) * self.audio_noise_std
            audio_np = audio_np + noise

        audio = self.aud_transform(audio_np)

        # ---- load image ----
        img_path = os.path.join(self.image_dir, self.image_files[img_idx])
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        return {
            "audio": audio,
            "image": image,
            "label": target_label,
        }


# ======================================================================
#  Convenience: pre‑built difficulty presets
# ======================================================================
CHALLENGE_PRESETS = {
    "clean": dict(
        marginal_mismatch=False,
        domain_shift=False,
        label_imbalance=False,
    ),
    "mild": dict(
        marginal_mismatch=True, marginal_ratio=0.7,
        domain_shift=True, domain_shift_level=0.3,
        label_imbalance=True, imbalance_factor=5.0,
    ),
    "moderate": dict(
        marginal_mismatch=True, marginal_ratio=0.5,
        domain_shift=True, domain_shift_level=0.5,
        label_imbalance=True, imbalance_factor=10.0,
    ),
    "hard": dict(
        marginal_mismatch=True, marginal_ratio=0.3,
        domain_shift=True, domain_shift_level=0.8,
        label_imbalance=True, imbalance_factor=50.0,
    ),
    # --- single‑challenge ablations ---
    "marginal_only": dict(
        marginal_mismatch=True, marginal_ratio=0.5,
        domain_shift=False,
        label_imbalance=False,
    ),
    "domain_only": dict(
        marginal_mismatch=False,
        domain_shift=True, domain_shift_level=0.5,
        label_imbalance=False,
    ),
    "imbalance_only": dict(
        marginal_mismatch=False,
        domain_shift=False,
        label_imbalance=True, imbalance_factor=10.0,
    ),
}
