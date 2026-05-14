# dataset settings
# PATH : /home/AD.UNLV.EDU/bhattb3/segmenter_semi_supervised/segm/data/config/meta_dataset.py

dataset_type = "MetaDataset"
data_root = "/home/AD.UNLV.EDU/bhattb3/Datasets/Meta/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
max_ratio = 4

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),  # <- important
    dict(type="Resize", img_scale=(512 * max_ratio, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512 * max_ratio, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512 * max_ratio, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,

    # ---------------------------
    # TRAIN
    # ---------------------------
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/train",
        ann_dir="masks/train",
        pipeline=train_pipeline,
    ),

    # ---------------------------
    # TRAIN + VAL COMBINED
    # ---------------------------
    trainval=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=["images/train", "images/validation"],
        ann_dir=["masks/train", "masks/validation"],
        pipeline=train_pipeline,
    ),

    # ---------------------------
    # VALIDATION
    # ---------------------------
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/validation",
        ann_dir="masks/validation",
        pipeline=val_pipeline,
    ),

    # ---------------------------
    # TEST (DEFAULT = META DATASET)
    # But can be overridden using argparse
    # ---------------------------
    test=dict(
        type=dataset_type,
        data_root=data_root,      # This will be changed by command line if needed
        img_dir="images/test",    # Also overridden if needed
        ann_dir="masks/test",
        pipeline=test_pipeline,
    ),
)
