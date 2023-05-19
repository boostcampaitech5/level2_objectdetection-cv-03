r"""
NOTE
    classes 설정
    wandb 설정
    test_mode 추가
"""

_base_ = [
    'cornernet_hourglass104_mstest_8x6_210e_coco.py',
    'coco_detection.py',
    'schedule_1x.py', 'default_runtime.py'
]