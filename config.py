_base_ = [
    '/content/mmclassification/configs/_base_/models/efficientnet_v2/efficientnetv2_l.py',
    '/content/mmclassification/configs/_base_/schedules/imagenet_bs256_coslr.py',
    '/content/mmclassification/configs/_base_/default_runtime.py'
]

# ---- model settings ----
# Here we use init_cfg to load pre-trained model.
# In this way, only the weights of backbone will be loaded.
# And modify the num_classes to match our dataset.

model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNetV2', arch='l'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=7,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# ---- data settings ----
# We re-organized the dataset as `CustomDataset` format.
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/content/drive/MyDrive/data/sapxeptrain',
        classes=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/content/drive/MyDrive/data/sapxepvalid',
        classes=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/content/drive/MyDrive/data/sapxeptest',
        classes=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Specify the evaluation metric for validation and testing.
val_evaluator = dict(type='Accuracy', topk=1)
test_evaluator = val_evaluator

# ---- schedule settings ----
# Usually in fine-tuning, we need a smaller learning rate and less training epochs.
# Specify the learning rate
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
# Set the learning rate scheduler
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(logger=dict(interval=100))

randomness = dict(seed=42, deterministic=False)