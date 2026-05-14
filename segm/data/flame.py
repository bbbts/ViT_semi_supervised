import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T

from segm.data import utils
from segm.config import dataset_dir

IGNORE_LABEL = 255

STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "default": {"mean": (127.5, 127.5, 127.5), "std": (127.5, 127.5, 127.5)},
}

FLAME_CONFIG_PATH = Path(__file__).parent / "config" / "flame.py"
FLAME_CATS_PATH = Path(__file__).parent / "config" / "flame.yml"

class FlameDataset(Dataset):
    def __init__(self, image_size=512, crop_size=512, split="train", normalization="vit", root=None, semi_supervised=False):
        self.root = Path(root or "/home/AD.UNLV.EDU/bhattb3/Datasets/Flame/")
        self.split = split
        self.semi_supervised = semi_supervised  # if True, some images can be unlabeled

        self.image_dir = self.root / "images" / self.split
        self.mask_dir = self.root / "masks" / self.split

        self.images = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        self.masks = sorted(list(self.mask_dir.glob("*.png")))

        if len(self.images) == 0:
            raise RuntimeError(f"No images found at {self.image_dir}")

        if not semi_supervised:
            if len(self.masks) == 0:
                raise RuntimeError(f"No masks found at {self.mask_dir}")
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

        self.n_cls = 2
        self.ignore_label = IGNORE_LABEL
        self.names, self.colors = utils.dataset_cat_description(FLAME_CATS_PATH)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform_img(img)
    
        # Default: unlabeled mask
        mask_tensor = torch.full((self.image_size, self.image_size), IGNORE_LABEL, dtype=torch.long)
        is_labeled = False
    
        # Check if a mask exists for this image
        if idx < len(self.masks):
            mask_path = self.masks[idx]
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                mask = self.transform_mask(mask)
                mask = np.array(mask, dtype=np.uint8)
    
                # Detect if the mask is "blank" (all IGNORE_LABEL)
                if np.all(mask == IGNORE_LABEL):
                    mask_tensor = torch.full((self.image_size, self.image_size), IGNORE_LABEL, dtype=torch.long)
                    is_labeled = False
                else:
                    mask = np.where(mask > 0, 1, 0).astype(np.uint8)
                    mask_tensor = torch.from_numpy(mask).long()
                    is_labeled = True
    
        image_id = os.path.basename(img_path).split('.')[0]
    
        return {
            "image": img,
            "mask": mask_tensor,
            "is_labeled": is_labeled,
            "id": image_id
        }


    def get_gt_seg_maps(self):
        gt_seg_maps = {}
        for img_path, mask_path in zip(self.images, self.masks):
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)
            gt_seg_maps[os.path.basename(img_path).split('.')[0]] = mask
        return gt_seg_maps
