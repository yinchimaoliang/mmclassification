_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/gleason_2019_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MulLabelLinearClsHead',
        wei_net_backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        pool_kernel=32,
        num_classes=1000,
        fc_in_channels=512,
        num_expert=3,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        mul_label_ind=[1, 2, 3]))

data = dict(
    train=dict(ann_file=[
        'data/gleason_2019/train.txt', 'data/gleason_2019/train_1.txt',
        'data/gleason_2019/train_3.txt', 'data/gleason_2019/train_4.txt'
    ]), )
