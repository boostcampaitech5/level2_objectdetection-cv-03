# optimizer
optimizer = dict(
    type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=0.001,
)
# runner = dict(type="IterBasedRunner", max_iters=1049 * 12)
runner = dict(type="EpochBasedRunner", max_epochs=20)
