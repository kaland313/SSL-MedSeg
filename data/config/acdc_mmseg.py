# MMSegmentation custom dataset with similar config as used for the acdc dataset
# https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.datasets.CustomDataset.
dataset_type = "CustomDataset"
data_root = "data/acdc/"
num_classes = 4 # Number of segmentation classes 
in_channels = 1  # Number of input channels (e.g. 3 for RGB data)
class_labels = ["BG", "RV", "MYO", "LV"]
ignore_label = 0 
img_suffix='.tiff'
seg_map_suffix = '_gt.tiff'
# img_norm_cfg = dict(
#     mean=[67.27297657740893], std=[84.6606962344396], to_rgb=True)
# crop_size = (224, 224)

# For transform pipeline docs see https://mmsegmentation.readthedocs.io/en/latest/api.html#module-mmseg.datasets.pipelines
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='LoadAnnotations'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
    dict(type='LoadAnnotations'),
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