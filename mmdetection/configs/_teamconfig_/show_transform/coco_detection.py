dataset_type = "CocoDataset"
data_root = "/opt/ml/dataset/"
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

train_pipeline = [
    dict(type="Mosaic", img_scale=(512,512), center_ratio_range=(0.5, 1.5), prob=0.5),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True), # Don't fix it
    dict(type="RandomFlip", flip_ratio=0.3, direction="horizontal"),# flip_ratio: 0.0 or value
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
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
            ann_file=data_root + "train2_swj.json",
            img_prefix=data_root,
            classes=classes,
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations", with_bbox=True),
            ],
        ),
    ),
)
