'''
Config Template

1. 

2.

'''

#-- Example

_base_ = [
    '../_teamconfig_/models/faster_rcnn_r50_fpn.py',
    '../_teamconfig_/datasets/coco_detection.py',
    '../_teamconfig_/schedules/schedule_1x.py', '../_teamconfig_/default_runtime.py'
]