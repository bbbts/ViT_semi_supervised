# PATH : /home/AD.UNLV.EDU/bhattb3/segmenter_semi_supervised/segm/data/meta.py

import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

# ----------------------------
# Add these lines for inference.py compatibility
# ----------------------------
from segm.data import utils
from segm.config import dataset_dir

META_CONFIG_PATH = Path(__file__).parent / "config" / "meta.py"
META_CATS_PATH = Path(__file__).parent / "config" / "meta.yml"

IGNORE_LABEL = 255

STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "default": {"mean": (127.5, 127.5, 127.5), "std": (127.5, 127.5, 127.5)},
}

# ---------------------------------------------------------
# Remap mask utility (keeps original labels if desired)
# ---------------------------------------------------------
def remap_mask(mask, label_map=None):
    if label_map is None:
        return mask  # keep original 0..n_cls-1 labels
    remapped = np.full_like(mask, fill_value=IGNORE_LABEL, dtype=np.int64)
    for k, v in label_map.items():
        remapped[mask == k] = v
    return remapped

class MetaDataset(Dataset):
    def __init__(self, image_size=512, crop_size=512, split="train", normalization="vit", root=None):
        self.root = Path(root or "/home/AD.UNLV.EDU/bhattb3/Datasets/Meta/")
        self.split = split
        self.image_dir = self.root / "images" / self.split
        self.mask_dir = self.root / "masks" / self.split

        self.images = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        self.masks = sorted(list(self.mask_dir.glob("*.png")))

        if len(self.images) == 0 or len(self.masks) == 0:
            raise RuntimeError(f"No images or masks found. Check paths: {self.image_dir}, {self.mask_dir}")
        if len(self.images) != len(self.masks):
            raise ValueError(f"Number of images ({len(self.images)}) and masks ({len(self.masks)}) do not match!")

        self.image_size = image_size
        self.crop_size = crop_size
        self.normalization = STATS.get(normalization, STATS["default"]).copy()

        self.transform_img = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.normalization["mean"], std=self.normalization["std"]),
        ])
        self.transform_mask = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.NEAREST)
        ])

        self.n_cls = 4
        self.ignore_label = IGNORE_LABEL
        self.reduce_zero_label = False

        # Add category names/colors
        self.names, self.colors = utils.dataset_cat_description(META_CATS_PATH)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        mask = np.array(mask, dtype=np.uint8)

        # Optional: remap labels according to meta.yml if needed
        label_map = {0: 0, 1: 1, 2: 2, 3: 3}
        mask = remap_mask(mask, label_map)

        image_id = os.path.basename(img_path).split('.')[0]
        return {"image": img, "mask": mask, "id": image_id}

    @property
    def dataset(self):
        return self

    def get_gt_seg_maps(self):
        gt_seg_maps = {}
        for img_path, mask_path in zip(self.images, self.masks):
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            # Optional: remap labels
            label_map = {0: 0, 1: 1, 2: 2, 3: 3}
            mask = remap_mask(mask, label_map)
            gt_seg_maps[os.path.basename(img_path).split('.')[0]] = mask
        return gt_seg_maps
