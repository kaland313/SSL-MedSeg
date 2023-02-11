# Self-Supervised Pretraining for 2D Medical Image Segmentation

This repository is the official implementation of [Self-Supervised Pretraining for 2D Medical Image Segmentation (accepted for the AIMIA workshop at ECCV 2022)](https://doi.org/10.1007/978-3-031-25082-8_31). 

![pretraining_strategies](.github/pretraining_strategies.svg)

If you use our code or results, please cite our paper: 

```
@InProceedings{Kalapos2022,
  author    = {Kalapos, Andr{\'a}s and Gyires-T{\'o}th, B{\'a}lint},
  booktitle = {Computer Vision -- ECCV 2022 Workshops},
  title     = {{Self-supervised Pretraining for 2D Medical Image Segmentation}},
  year      = {2023},
  address   = {Cham},
  pages     = {472--484},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-25082-8_31},
  isbn      = {978-3-031-25082-8},
}

```



## Requirements

### Required python packages

To install pypi requirements:

```setup
pip install -r requirements.txt
```

For self-supervised pre-training `solo-learn==1.0.5` is also required. For it's installation, follow instructions in [solo-learn's documentation](https://solo-learn.readthedocs.io/en/latest/start/install.html) (dali, umap support is not needed) or use the following commands: 

```
git clone https://github.com/vturrisi/solo-learn.git
cd solo-learn
git checkout 1.0.5
pip3 install -e .
```

### Dataset setup

Download the ACDC Segmentation dataset from: https://acdc.creatis.insa-lyon.fr (registration required)

Specify the path for the dataset in:

- [supervised_segmentation/config_acdc.yaml](supervised_segmentation/config_acdc.yaml) -> `dataset_root:  "<YOUR DATASET PATH>"`
- [self_supervised/byol-acdc.sh](self_supervised/byol-acdc.sh) -> `--data_dir  <YOUR DATASET PATH>`

### Slurm

For hyperparamter sweeps and running many experiments in a batch, we use [Slurm](https://slurm.schedmd.com/documentation.html) jobs, therefore an installed and configured Slurm environment is required for these runs. However based on [supervised_segmentation/sweeps/data_eff_learning.sh](supervised_segmentation/sweeps/data_eff_learning.sh) and [supervised_segmentation/sweeps/grid_search_helper.py](supervised_segmentation/sweeps/grid_search_helper.py) other methods of running experiments in a batch can be implemented. 



## Training

### Supervised segmentation (downstream) training 

To train the model(s) in the paper, run this command: 

```
PYTHONPATH=. python supervised_segmentation/train_acdc.py
```

On a Slurm cluster:

```
PYTHONPATH=. srun -p gpu --gres=gpu --cpus-per-task=10 python supervised_segmentation/train_acdc.py
```

To initialize the downstream training with different pretrained models, we provide pretrained weights that we used in our paper. These can be selected by setting the `encoder_weights` config in [supervised_segmentation/config_acdc.yaml](supervised_segmentation/config_acdc.yaml)

| Pre-training approach           | Corresponding arrow on the figure above                      | `encoder_weights`                                            |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Supervised ImageNet             | ![arrow-generalist-supervised](.github/arrow-generalist-supervised.svg) | `supervised_imagenet`                                        |
| BYOL ImageNet                   | ![arrow-generalist-selfsupervised](.github/arrow-generalist-selfsupervised.svg) | `resnet50_byol_imagenet2012.pth.tar`                         |
| Supervised ImageNet + BYOL ACDC | ![arrow-hierarchical-supervised](.github/arrow-hierarchical-supervised.svg) (2nd step) | `supervised-imagenet-byol-acdc-ep=25.pth` |
| BYOL ImageNet + BYOL ACDC       | ![arrow-hierarchical-selfsupervised](.github/arrow-hierarchical-selfsupervised.svg) (2nd step) | `byol-imagenet-acdc-ep=34.ckpt`<br />or <br />`byol-imagenet-acdc-ep=25.pth` |
| BYOL ACDC                       | ![arrow-specialist](.github/arrow-specialist.svg)            | `byol_acdc_backbone_last.pth`                                |

### Pretraining

For ImageNet pretraining, we acquire weights from:

- [`timm`](https://github.com/rwightman/pytorch-image-models) for supervised ImageNet pretraining
- [github.com/yaox12/BYOL-PyTorch](https://github.com/yaox12/BYOL-PyTorch) for BYOL ImageNet pretraing [[Google drive link for their model file](https://drive.google.com/file/d/1TLZHDbV-qQlLjkR8P0LZaxzwEE6O_7g1/view?usp=sharing)]

To pretrain a model on the ACDC dataset, run this command: 

```
self_supervised/byol-acdc.sh
```

Different pretraining strategies can be configured by modifying the value for the  `--pretrained_weights` and `--max_epochs` configs in `self_supervised/byol-acdc.sh`

| Pre-training approach           | Corresponding arrow on the figure above                      | `--pretrained_weights`               | `--max_epochs` | Published pretrained model <br /> [[models.zip](https://github.com/kaland313/SSL-MedSeg/releases/download/v1.0/models.zip)]|
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------ | -------------- | ------------------------------------------------------------ |
| Supervised ImageNet + BYOL ACDC | ![arrow-hierarchical-supervised](.github/arrow-hierarchical-supervised.svg)(1st step) | `supervised_imagenet`                | 25             | `models/supervised-imagenet-byol-acdc-ep=25.pth` |
| BYOL ImageNet + BYOL ACDC       | ![arrow-hierarchical-selfsupervised](.github/arrow-hierarchical-selfsupervised.svg) (1st step) | `resnet50_byol_imagenet2012.pth.tar` | 25             | `models/byol-imagenet-acdc-ep=34.ckpt`<br />and  <br />`models/byol-imagenet-acdc-ep=25.pth` |
| BYOL ACDC                       | ![arrow-specialist](.github/arrow-specialist.svg)            | `None`                               | 400            | `models/byol_acdc_backbone_last.pth`                         |

We publish pretrained models for these pretrainings as specified in the last column of the table

## Evaluation

To evaluate the segmentation model on the ACDC dataset, run:

```eval
PYTHONPATH=. python supervised_segmentation/train_acdc.py
```

## Copyright 

Segmentation code is based on: [<img src="https://github.githubassets.com/pinned-octocat.svg" style="height:14pt;" />ternaus/cloths_segmentation](https://github.com/ternaus/cloths_segmentation)

Self-supervised training script is based on: [<img src="https://github.githubassets.com/pinned-octocat.svg" style="height:14pt;" />vturrisi/solo-learn](https://github.com/vturrisi/solo-learn)

BYOL ImageNet pretrained model from: [<img src="https://github.githubassets.com/pinned-octocat.svg" style="height:14pt;" />yaox12/BYOL-PyTorch](https://github.com/yaox12/BYOL-PyTorch) [[Google drive link for their model file](https://drive.google.com/file/d/1TLZHDbV-qQlLjkR8P0LZaxzwEE6O_7g1/view?usp=sharing)]

This Readme is based on: [<img src="https://github.githubassets.com/pinned-octocat.svg" style="height:14pt;" />paperswithcode/releasing-research-code]( https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md)

