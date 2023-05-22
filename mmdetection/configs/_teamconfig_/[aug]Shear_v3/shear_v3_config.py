r"""
NOTE
    classes 설정
    wandb 설정
    test_mode 추가
"""

_base_ = [
    'faster_rcnn_r50_fpn.py',
    'coco_detection.py',
    'schedule_1x.py', 'default_runtime.py'
]
