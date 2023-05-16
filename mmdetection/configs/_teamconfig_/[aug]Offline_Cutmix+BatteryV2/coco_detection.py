dataset_type = "CocoDataset"
data_root = "../../../dataset/swj_cutmix_bupsample/"
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
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),  # Don't fix it
    # dict(type=""),  # Fill augmentation method
    dict(type='Mosaic', img_scale=(512, 512),
            center_ratio_range=(1, 1),
            prob=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(
                type="ElasticTransform",
                p=0.7,
                alpha=35, 
                sigma=10, 
                alpha_affine=150
            )
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
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
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
# NOTE: classes 설정
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type="MultiImageMixDataset",
        pipeline=train_pipeline,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "swj_cutmix_battery_train.json",
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
        ann_file=data_root + "swj_cutmix_battery_val.json",
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
