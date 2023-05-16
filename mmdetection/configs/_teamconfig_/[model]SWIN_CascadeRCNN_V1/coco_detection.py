dataset_type = "CocoDataset"
data_root = "../../../dataset/swj_cutmix_battery/"
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

a_transform = [
    dict(type='RandomRotate90', p=0.5),
    dict(type='MedianBlur', blur_limit=3, p=0.2)
]

# Default Setting (Resize, RandomFlip, Normalize, Pad, DefaultFormatBundle, Collect)
img_scale = [(512, 512),(768,768), (1024,1024)]
train_pipeline = [
    dict(type="Resize", img_scale=img_scale, multiscale_mode='value', keep_ratio=True),  # Don't fix it
    # dict(type=""),  # Fill augmentation method
    dict(type='Mosaic', img_scale=img_scale, multiscale_mode='value',
            center_ratio_range=(0.5, 1.0),
            prob=0.3),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Albu', transforms=a_transform, 
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
                'img': 'image',
                'gt_bboxes': 'bboxes'
                },
        update_pad_shape=False,
        skip_img_without_anno=True
        ),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.2],
        contrast_limit=[0.1, 0.2],
        p=0.3),
    
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="MultiImageMixDataset",
        pipeline=train_pipeline,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "train.json",
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
        ann_file=data_root + "val.json",
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
