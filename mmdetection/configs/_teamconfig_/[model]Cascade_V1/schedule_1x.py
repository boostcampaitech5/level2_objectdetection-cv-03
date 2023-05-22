# optimizer
optimizer = dict(type="Adam", lr=0.02, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000, step=[8, 11])
lr_config = dict(
    policy="CosineAnnealing", warmup="linear", warmup_iters=500, warmup_ratio=0.001,min_lr_ratio=0.001
)
runner = dict(type="EpochBasedRunner",max_epochs=12)
