dataset_type = "CocoDataset"
data_root = "../../../dataset/"
classes = (
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
# Default Setting (Resize, RandomFlip, Normalize, Pad, DefaultFormatBundle, Collect)
train_pipeline = [
    dict(
        type="Resize", img_scale=(1024, 1024), ratio_range=(0.2, 1.0), keep_ratio=True
    ),  # Don't fix it
    # dict(type=""),  # Fill augmentation method
    dict(
        type="RandomFlip", flip_ratio=0.3, direction="horizontal"
    ),  # flip_ratio: 0.0 or value
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=[(1024, 1024), (512, 512), (256, 256)],
        flip=True,
        transforms=[
            dict(type="Resize", multiscale_mode="value", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="PhotoMetricDistortion"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=[(1024, 1024), (512, 512), (256, 256)],
        flip=True,
        transforms=[
            dict(type="Resize", multiscale_mode="value", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="PhotoMetricDistortion"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
# NOTE: classes 설정
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="MultiImageMixDataset",
        pipeline=train_pipeline,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "train_kjy.json",
            img_prefix=data_root,
            classes=classes,
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations", with_bbox=True),
            ],
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val_kjy.json",
        img_prefix=data_root,
        pipeline=val_pipeline,
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
        # NOTE test_mode 추가
        test_mode=True,
    ),
)
evaluation = dict(interval=1, metric="bbox")
