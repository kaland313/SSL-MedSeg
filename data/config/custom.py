# MMSegmentation custom dataset with similar config as used for the acdc dataset
# https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.datasets.CustomDataset.
dataset_type = "CustomDataset"
data_root = PATH_TO_DATASET # Path to custom dataset
num_classes = NUM_CLASSES # Number of segmentation classes 
in_channels = IN_CHANNELS  # Number of input channels (e.g. 3 for RGB data)
class_labels = CLASS_LABELS # Class labels used for logging
ignore_label = IGNORE_LABEL  # Ignored label during iou metric computation
img_suffix='.tiff'
seg_map_suffix = '_gt.tiff'
img_norm_cfg = dict(
    mean=[67.27297657740893], std=[84.6606962344396], to_rgb=True)
crop_size = (224, 224)

# For transform pipeline docs see https://mmsegmentation.readthedocs.io/en/latest/api.html#module-mmseg.datasets.pipelines
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(224, 224), ratio_range=(0.75, 1.3333)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=ignore_label),
    # dict(type='DefaultFormatBundle'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(2048, 1024),  keep_ratio=True),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='DefaultFormatBundle'),
]

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pipeline=test_pipeline))