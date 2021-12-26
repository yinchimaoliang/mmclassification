# dataset settings
dataset_type = 'Glomerulus'
img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)
train_pipeline = [
    dict(type='Resize', size=(64, 64)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=(64, 64)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/glomerulus/train',
        ann_file='data/glomerulus/train/labels.pkl',
        classes=[
            'GS', 'SS+M', 'M', 'GM', 'N', 'C', 'SS', 'E+M+SS', 'E+SS', 'C+M',
            'E', 'C+E'
        ],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/glomerulus/val',
        ann_file='data/glomerulus/val/labels.pkl',
        pipeline=test_pipeline,
        classes=[
            'GS', 'SS+M', 'M', 'GM', 'N', 'C', 'SS', 'E+M+SS', 'E+SS', 'C+M',
            'E', 'C+E'
        ],
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/glomerulus/val',
        ann_file='data/glomerulus/val/labels.pkl',
        pipeline=test_pipeline,
        classes=[
            'GS', 'SS+M', 'M', 'GM', 'N', 'C', 'SS', 'E+M+SS', 'E+SS', 'C+M',
            'E', 'C+E'
        ],
        test_mode=True))
