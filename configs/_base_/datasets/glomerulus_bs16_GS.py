# dataset settings
dataset_type = 'Glomerulus'
img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)
train_pipeline = [
    dict(type='Resize', size=(1024, 1024)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=(1024, 1024)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/glomerulus/universal/train/GS',
        ann_file='data/glomerulus/universal/train/GS/labels.pkl',
        classes=[
            'background', 'GS'
        ],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/glomerulus/universal/val/GS',
        ann_file='data/glomerulus/universal/val/GS/labels.pkl',
        pipeline=test_pipeline,
        classes=[
            'background', 'GS'
        ],
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/glomerulus/universal/val/GS',
        ann_file='data/glomerulus/universal/val/GS/labels.pkl',
        pipeline=test_pipeline,
        classes=[
            'background', 'GS'
        ],
        test_mode=True))
