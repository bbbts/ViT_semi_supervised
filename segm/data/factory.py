# PATH : /home/AD.UNLV.EDU/bhattb3/segmenter_supervised/segm/data/factory.py

from segm.data.flame import FlameDataset
from segm.data.ade20k import ADE20KSegmentation
from segm.data.pascal_context import PascalContextDataset
from segm.data.cityscapes import CityscapesDataset
from segm.data.meta import MetaDataset  # <--- NEW
import random
import numpy as np
from PIL import Image
import os

IGNORE_LABEL = 255


def create_dataset(kwargs):
    """
    kwargs can include:
    - dataset: name of dataset
    - image_size, crop_size, split, normalization, root
    - labeled_ratio (float, optional): for semi-supervised training
    """

    dataset_name = kwargs.get("dataset", "").lower()
    split = kwargs.get("split", "train").lower()  # normalize to lowercase
    labeled_ratio = kwargs.get("labeled_ratio", 1.0)  # default = fully labeled

    # Accept synonyms for training split
    if split in ["training", "train"]:
        split_normalized = "train"
    elif split in ["validation", "val", "validate"]:
        split_normalized = "val"
    elif split in ["test", "testing"]:
        split_normalized = "test"
    else:
        split_normalized = split  # keep other names as is

    # -----------------------------
    # Dataset selection
    # -----------------------------
    if dataset_name == "flame":
        dataset = FlameDataset(
            image_size=kwargs.get("image_size", 512),
            crop_size=kwargs.get("crop_size", 512),
            split=split_normalized,
            normalization=kwargs.get("normalization", "vit"),
            root=kwargs.get("root", "/home/AD.UNLV.EDU/bhattb3/Datasets/Flame"),
        )

    elif dataset_name == "ade20k":
        dataset = ADE20KSegmentation(**kwargs)

    elif dataset_name == "pascal":
        dataset = PascalContextDataset(**kwargs)

    elif dataset_name == "cityscapes":
        dataset = CityscapesDataset(**kwargs)

    elif dataset_name == "meta":
        dataset = MetaDataset(
            image_size=kwargs.get("image_size", 512),
            crop_size=kwargs.get("crop_size", 512),
            split=split_normalized,
            normalization=kwargs.get("normalization", "vit"),
            root=kwargs.get("root", "/home/AD.UNLV.EDU/bhattb3/Datasets/Meta"),
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # -------------------------------
    # Apply semi-supervised mask blanking if training
    # -------------------------------
    if split_normalized == "train" and labeled_ratio < 1.0:
        n_samples = len(dataset)
        n_labeled = int(n_samples * labeled_ratio)
        labeled_indices = set(random.sample(range(n_samples), n_labeled))

        print(f"Semi-supervised mode: {n_labeled}/{n_samples} labeled, {n_samples-n_labeled} blank masks.")

        # Patch the dataset __getitem__ method dynamically
        original_getitem = dataset.__getitem__

        def semi_supervised_getitem(idx):
            item = original_getitem(idx)

            # Determine original mask key
            mask_key = "segmentation" if "segmentation" in item else ("mask" if "mask" in item else None)
            mask = item.get(mask_key, None)

            if idx not in labeled_indices:
                # Replace mask with blank mask (preserve type: numpy array or PIL)
                if isinstance(mask, np.ndarray):
                    blank_mask = np.full_like(mask, IGNORE_LABEL)
                elif hasattr(mask, "size"):  # PIL Image
                    blank_mask = Image.fromarray(np.full((mask.height, mask.width), IGNORE_LABEL, dtype=np.uint8))
                else:
                    blank_mask = np.array([[IGNORE_LABEL]], dtype=np.uint8)

                if mask_key == "segmentation":
                    item["segmentation"] = blank_mask
                elif mask_key == "mask":
                    item["mask"] = blank_mask
                else:
                    item["mask"] = blank_mask

                item["is_labeled"] = False
            else:
                item["is_labeled"] = True

            return item

        dataset.__getitem__ = semi_supervised_getitem

    else:
        # Default: mark all as labeled
        original_getitem = dataset.__getitem__

        def add_flag_getitem(idx):
            item = original_getitem(idx)
            item["is_labeled"] = True
            return item

        dataset.__getitem__ = add_flag_getitem

    return dataset
