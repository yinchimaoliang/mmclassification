_base_ = [
    '../_base_/models/resnet18_cifar.py',
    '../_base_/datasets/glomerulus_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=152,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=12,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
evaluation = dict(interval=1, save_best="auto", metric='accuracy', metric_options={'topk': (1, )})
checkpoint_config = dict(interval=-1, save_last=True)