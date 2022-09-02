# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from statistics import mode
import time
from pprint import pprint

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

import solo
from solo.args.setup import parse_args_pretrain
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer

try:
    from solo.methods.dali import PretrainABC
except ImportError as e:
    print(e)
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

import types

from solo.utils.checkpointer import Checkpointer
from solo.utils.classification_dataloader import prepare_data as prepare_data_classification
from solo.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
    dataset_with_index
)

import torch
import hashlib
import albumentations as A
from argparse import ArgumentParser
from data.acdc_dataset import ACDCDatasetAlbu, ACDCDatasetUnlabeleld, DATASET_MEAN, DATASET_STD
from self_supervised.transforms import AlbuTransforms
from self_supervised.weights_saver import WeightsSaver
from models.pretrained_models import get_state_dict_form_pretrained_model_zoo
import utils

def add_custom_args(parser: ArgumentParser):
    parser.add_argument("--pretrained_weights", type=str, default=None, 
                        help="Path of base ptretrainign to be loaded as initial weights")
    return parser

def main():
    seed_everything(5)

    # scriptd_parser = ArgumentParser()
    # arg = script_parser.add_argument
    # arg("--load_weights", type=str, default="", help=)
    # script_args = script_parser.parse_known_args()

    args = parse_args_pretrain(add_custom_args)
    print(utils.pretty_print(args.__dict__))

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    if args.num_large_crops != 2:
        assert args.method == "wmse"

    MethodClass = METHODS[args.method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = types.new_class(f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))

    # Initialize model and backbone
    in_chans = 1
    args.backbone_args["in_chans"] = in_chans  # for use with timm resnets (maybe transformers as well)
    args.backbone_args.pop("zero_init_residual", None) # for use with timm resnets (maybe transformers as well)
    if args.pretrained_weights == 'supervised_imagenet':
        args.backbone_args["pretrained"] = True
    model = MethodClass(**args.__dict__)

    # Load pretrained weights 
    if args.pretrained_weights not in [None, 'None', 'supervised_imagenet']: 
        pretrained_weights = get_state_dict_form_pretrained_model_zoo(args.pretrained_weights, in_chans)
        model.backbone.load_state_dict(pretrained_weights)
        print("Encoder pretrained weights loaded")
        if isinstance(model, solo.methods.base.BaseMomentumMethod):
            model.momentum_backbone.load_state_dict(pretrained_weights)
            print("Momentum Encoder pretrained weights loaded")
    
    backbone_hash = hashlib.md5(str(dict(model.backbone.state_dict())).encode()).hexdigest()
    print("Backbone state_dict hash:",  backbone_hash)  
    
    # pretrain dataloader
    if not args.dali:
        # asymmetric augmentations
        if args.unique_augs > 1:
            transform = [
                AlbuTransforms(**kwargs) for kwargs in args.transform_kwargs
            ]
        else:
            transform = [AlbuTransforms(**args.transform_kwargs)],
                
        transform = prepare_n_crop_transform(transform, num_crops_per_aug=args.num_crops_per_aug)
        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        # train_dataset = prepare_datasets(
        #     args.dataset,
        #     transform,
        #     data_dir=args.data_dir,
        #     train_dir=args.train_dir,
        #     no_labels=args.no_labels,
        # )
        # train_dataset = ACDCDatasetAlbu(args.data_dir, transform)
        train_dataset = dataset_with_index(ACDCDatasetUnlabeleld)(args.data_dir, transform)
        train_loader = prepare_dataloader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )

        start = time.time()
        test_batch = next(iter(train_loader))
        batch_time = (time.time()-start)
        print(f"Generating a batch took {batch_time:.2f} s. Per-sample generation time: {batch_time/args.batch_size:0.3f}s")
        print(f"Batch_size = {args.batch_size}; Num_crops = {len(test_batch)}")


    # normal dataloader for when it is available
    if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
        val_loader = None
    elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
        val_loader = None
    else:
        _, val_loader = prepare_data_classification(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    callbacks = []

    loggers = []
    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            offline=args.offline,
        )
        # wandb_logger.watch(model, log="gradients", log_freq=100)
        
        wandb_logger.log_hyperparams(args)
        wandb_logger.log_hyperparams({ "backbone_hash": backbone_hash})

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        loggers.append(wandb_logger)
    
    if args.save_checkpoint:
        # save checkpoint on last epoch only
        logdir = os.path.join(args.checkpoint_dir, args.method)
        print("Checkpoint dir: ", logdir)
        ckpt = Checkpointer(
            args,
            logdir=logdir,
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)
    
    weights_saver = WeightsSaver(model, list(range(10)) + list(range(10, 25, 2)) + [25, 50, 100, 200, 300, 400])
    callbacks.append(weights_saver)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path = None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, args.method),
            max_hours=args.auto_resumer_max_hours,
        )
        resume_from_checkpoint = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    if not torch.cuda.is_available():
        args.gpus = 0
        args.accelerator = 'cpu'
    trainer = Trainer.from_argparse_args(
        args,
        logger=loggers if loggers != [] else None,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=True) if args.accelerator == "ddp" else None,
        enable_checkpointing=False,
    )

    if args.dali:
        trainer.fit(model, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    torch.save(model.backbone.state_dict(), os.path.join(ckpt.path, 'backbone_last.pth'))

if __name__ == "__main__":
    main()