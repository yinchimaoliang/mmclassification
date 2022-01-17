# dataset settings
dataset_type = 'Glomerulus'
img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)
train_pipeline = [
    dict(type='Resize', size=(256, 256)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/glomerulus/instance/bin/train/SS',
        ann_file='data/glomerulus/instance/bin/train/SS/labels.pkl',
        classes=['background', 'SS'],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/glomerulus/instance/bin/val/SS',
        ann_file='data/glomerulus/instance/bin/val/SS/labels.pkl',
        pipeline=test_pipeline,
        classes=['background', 'SS'],
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/glomerulus/instance/bin/val/SS',
        ann_file='data/glomerulus/instance/bin/val/SS/labels.pkl',
        pipeline=test_pipeline,
        classes=['background', 'SS'],
        test_mode=True))