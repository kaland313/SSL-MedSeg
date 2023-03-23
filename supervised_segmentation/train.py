import argparse
import os
from pathlib import Path
from typing import Dict
import yaml
import numpy as np
from tabulate import tabulate

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from albumentations.core.serialization import from_dict
import albumentations as A
import segmentation_models_pytorch as smp
import torchmetrics
import wandb

from data.mmseg_dataset import MMseg
from data.acdc_dataset import ACDCDatasetAlbu, DATASET_MEAN, DATASET_STD
from models.pretrained_models import get_state_dict_form_pretrained_model_zoo, modify_state_dict, pretrained_model_zoo
import utils


class CardiacSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        if self.config["dataset"].lower() == "acdc":
            self.num_classes = 4
            self.class_labels = ["BG", "RV", "MYO", "LV"]
            self.in_channels = 1
            self.ignore_index = 0 # WARN: This is only used when computing iou metrics
        else:
            dataset_config = MMseg.get_config(self.config["dataset"])
            self.num_classes = dataset_config.num_classes
            self.in_channels = dataset_config.in_channels
            self.class_labels = dataset_config.class_labels
            # WARN: ignore_index is only used when computing iou metrics
            self.ignore_index = dataset_config.get("ignore_label", None) 

        # Instantiating encoder and loading pretrained weights
        encoder_weights = self.config["model"].get("encoder_weights", None)
        
        # Only supervised ImageNet weights can be loaded when instantiating an smp.Unet:
        if encoder_weights in ['imagenet', 'supervised-imagenet']:
            auto_loaded_encoder_weights = 'imagenet'
        else:
            auto_loaded_encoder_weights = None
            
        self.model =smp.Unet(
            encoder_name=self.config["model"]["encoder_name"],      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=auto_loaded_encoder_weights,            # for timm models, anything that is Not none will load imagenet weights...
            in_channels=self.in_channels,                           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.num_classes,                               # model output channels (number of classes in your dataset)
        )

        
        pretrained_weights = get_state_dict_form_pretrained_model_zoo(self.config["model"]["encoder_weights"], 
                                                                      in_chans = self.in_channels ,
                                                                      prefix_to_add='model.'
                                                                     )
        if pretrained_weights is not None:                                                            
            self.model.encoder.load_state_dict(pretrained_weights)
            print("Encoder pretrained weights loaded from {}".format(encoder_weights))

        if self.config["model"].get("pretrained_decoder", False):
            pretrained_decoder_with_head = torch.load("models/supervised_decoder.pth")
            pretrained_decoder = modify_state_dict(pretrained_decoder_with_head, prefix_to_remove='model.decoder.')
            pretrained_segmentation_head = modify_state_dict(pretrained_decoder_with_head, prefix_to_remove='model.segmentation_head.')
            self.model.decoder.load_state_dict(pretrained_decoder)
            self.model.segmentation_head.load_state_dict(pretrained_segmentation_head)
            print('Pretained weights for decoder loaded.')
        
        self.loss = smp.losses.__dict__[self.config["loss"]](smp.losses.MULTICLASS_MODE)
        self.val_focal_loss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
        
        self.train_iou = torchmetrics.JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index) 
        self.val_iou = torchmetrics.JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index)  
        self.best_val_iou = 0.    

        self.example_input_array = torch.zeros((1, 1,224,224))
    

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(batch)

    def setup(self, stage=0):
        if self.config["dataset"].lower() == "acdc":
            self.train_dataset = ACDCDatasetAlbu(self.config["dataset_root"], subset_ratio=self.config["subset_ratio"], oversamle=True)
            self.val_dataset = ACDCDatasetAlbu(self.config["dataset_root"], split='val')
            self.test_dataset = ACDCDatasetAlbu(self.config["dataset_root"], split='test')
        else:
            self.train_dataset = MMseg("train", self.config["dataset"])
            self.val_dataset = MMseg("val", self.config["dataset"])
            self.test_dataset = MMseg("test", self.config["dataset"])
        print("Number of  train samples = ", len(self.train_dataset))
        print("Number of val samples = ", len(self.val_dataset))
        print("Number of test samples = ", len(self.test_dataset))

    def train_dataloader(self):
        train_aug = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(DATASET_MEAN,), std=(DATASET_STD,)),
        ])
        self.train_dataset.transforms = train_aug

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(train_loader), "Train samples =", len(self.train_dataset))
        wandb.config.update({"num_samples": len(self.train_dataset)})
        return train_loader

    def val_dataloader(self):
        val_aug = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.Normalize(mean=(DATASET_MEAN,), std=(DATASET_STD,)),
        ])
        self.val_dataset.transforms = val_aug

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        print("Val dataloader = ", len(val_loader))
        return val_loader

    def configure_optimizers(self):
        optimizer = utils.object_from_dict(
            self.config["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = utils.object_from_dict(self.config["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        features, targets = batch
        targets = targets.to(torch.int64)

        logits = self.forward(features)
        total_loss = self.loss(logits, targets)

        self.log("train_loss", total_loss)
        self.log("train_iou", self.train_iou(preds=logits, target=targets.to(torch.int64)), on_epoch=True)
        self.log("lr",  self._get_current_lr())

        return total_loss

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0]

    def validation_step(self, batch, batch_id):
        features, targets = batch
        targets = targets.to(torch.int64)

        logits = self.forward(features)
        
        loss = self.loss(logits,targets)
        # "val_loss" and "val_iou" logged to allow easy comparison with older runs
        self.log("val_loss", loss) 
        val_iou = self.val_iou(preds=logits, target=targets)
        self.log("val_iou", val_iou)
        self.log("val_metrics/loss", loss)
        self.log("val_metrics/mean_iou", val_iou)
        if val_iou > self.best_val_iou: 
            self.best_val_iou = val_iou
        self.log("val_metrics/best_mean_iou", self.best_val_iou)
        self.log("val_metrics/focal_loss", self.val_focal_loss(logits, targets))
        per_class_ious = torchmetrics.functional.jaccard_index(logits, targets, absent_score=np.NaN, 
                                                               num_classes=self.num_classes, average='none')
        for i in range(self.num_classes):
            self.log(f"val_metrics/{self.class_labels[i]}_iou", per_class_ious[i])

        # if batch_id==0:
        #     class_labels_dict = {id:label for id, label in enumerate(self.class_labels)}

        #     def wb_mask(bg_img, pred_mask, true_mask):
        #         return wandb.Image(bg_img, masks={
        #             "prediction" : {"mask_data" : pred_mask, "class_labels" : class_labels_dict},
        #             "ground truth" : {"mask_data" : true_mask, "class_labels" : class_labels_dict}})

        #     bg_img = features[0, 0].detach().cpu().numpy()
        #     bg_img -= bg_img.min()
        #     bg_img /= bg_img.max()
        #     bg_img = np.expand_dims(bg_img, -1)
        #     pred_mask = torch.argmax(logits[0].detach(), dim=0).cpu().numpy().astype(np.uint8)
        #     true_mask = targets[0].detach().cpu().numpy().astype(np.uint8)
        #     wandb.log({"example_outputs": wb_mask(bg_img, pred_mask, true_mask)})

        return loss

    def on_after_backward(self) -> None:
        # Cancel gradients for the encoder in the first few epochs
        if self.current_epoch < self.config['model'].get("freeze_encoder_weights_epochs", 0):
            for p in self.model.encoder.parameters():
                p.grad = None
        return super().on_after_backward()

def main():
    default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config_acdc.yaml') 
    config = utils.get_config(default_config_path)

    pl.seed_everything(config["seed"], workers=True)

    pipeline = CardiacSegmentation(config)

    tb_logger = pl.loggers.TensorBoardLogger(config["artifacts_root"], name=config["experiment_name"], log_graph=False)
    wandb_logger = pl.loggers.WandbLogger(name=config["experiment_name"], project='ACDC-Segmentation', save_dir=config["artifacts_root"])
    #pl.LightningModule.hparams is set by pytorch lightning when calling save_hyperparameters
    tb_logger.log_hyperparams(pipeline.hparams)
    wandb_logger.log_hyperparams(pipeline.hparams)
    if config.get("wandb_tag", "") != "":
        wandb.run.tags = wandb.run.tags + (config["wandb_tag"],)


    checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            save_last=False, # False to reduce disk load from constant checkpointing
            save_top_k=1,
            monitor='val_metrics/mean_iou',
            mode='max',
            every_n_epochs=10,
        )
    # model_summary = pl.callbacks.ModelSummary(max_depth=5)

    trainer = pl.Trainer(gpus=0 if config["trainer"]["gpus"] == '0' or not torch.cuda.is_available() else config["trainer"]["gpus"],
                        max_epochs=config["trainer"]["max_epochs"],
                        precision=config["trainer"]["precision"] if torch.cuda.is_available() else 32,
                        logger=[tb_logger, wandb_logger],
                        callbacks=[checkpoint],
                        log_every_n_steps=20,
                        gradient_clip_val = config["trainer"]["gradient_clip_val"]
                        )

    trainer.fit(pipeline)

    print("=============================")
    print("Running predictions on the test dataset with the best model ")
    from supervised_segmentation.inference_acdc import generate_prediction_pdfs
    mean_iou, std_iou, per_class_ious, mean_loss, std_loss = generate_prediction_pdfs(checkpoint_path=checkpoint.best_model_path, 
                                                                                      output_folder=tb_logger.log_dir,
                                                                                      remove_individual_pdfs=True)
    headers = ["Resolution", "Mean IoU", "Std IoU", "BG IoU", "RV IoU", "MYO IoU", "LV IoU"]
    results = [["", mean_iou, std_iou, *per_class_ious]]
    print(tabulate(results, headers, tablefmt="pipe"))

    wandb.log({"Test/Mean IoU": mean_iou,
               "Test/St. dev. IoU": std_iou,
               "Test/BG IoU": per_class_ious[0], 
               "Test/RV IoU": per_class_ious[1], 
               "Test/MYO IoU": per_class_ious[2], 
               "Test/LV IoU": per_class_ious[3],
               "Test/Mean Loss": mean_loss,
               "Test/St. dev. Loss": std_loss})

if __name__ == "__main__":
    main()
