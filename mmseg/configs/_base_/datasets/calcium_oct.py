# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = '/home/gridsan/nchutisilp/datasets/CalciumOCT_VOC'
metainfo = dict(
    classes=('background', 'calcium'), 
    palette=[[0, 0, 0], [255, 0, 0]]
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False), #  If dataset only has two types of labels (i.e., label 0 and 1), it needs to close reduce_zero_label, i.e., set reduce_zero_label=False.
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=dict(img=0, seg=255)),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False), #  If dataset only has two types of labels (i.e., label 0 and 1), it needs to close reduce_zero_label, i.e., set reduce_zero_label=False.
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        reduce_zero_label=False, #  If dataset only has two types of labels (i.e., label 0 and 1), it needs to close reduce_zero_label, i.e., set reduce_zero_label=False.
        data_root=data_root,
        metainfo=metainfo,
        data_prefix = dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/all.txt',
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        reduce_zero_label=False, #  If dataset only has two types of labels (i.e., label 0 and 1), it needs to close reduce_zero_label, i.e., set reduce_zero_label=False.
        data_root=data_root,
        metainfo=metainfo,
        data_prefix = dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/train.txt',
        pipeline=test_pipeline)
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        reduce_zero_label=False, # If dataset only has two types of labels (i.e., label 0 and 1), it needs to close reduce_zero_label, i.e., set reduce_zero_label=False.
        data_root=data_root,
        metainfo=metainfo,
        data_prefix = dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/test.txt',
        pipeline=test_pipeline)
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator
