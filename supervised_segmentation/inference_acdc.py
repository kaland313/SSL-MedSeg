import os
import numpy as np
import cv2
from tqdm import tqdm
from tabulate import tabulate
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset
import albumentations as A
import torchmetrics
from tabulate import tabulate

import data.acdc_utils as acdc_utils
from data.acdc_dataset import ACDCDatasetAlbu, DATASET_MEAN, DATASET_STD
from supervised_segmentation.train import CardiacSegmentation
import utils

PRED_FOLDER = "inference"

def generate_prediction_pdfs(checkpoint_path: str,
                             dataset_root = "~/data/acdc/training",
                             num_samples = 10000,
                             output_folder=".",
                             merge_prediction_pdfs=True,
                             remove_individual_pdfs=False,
                             pad_to = 224):

    model = CardiacSegmentation.load_from_checkpoint(checkpoint_path)
    model.eval()

    test_augs = []
    if pad_to is not None:
        test_augs.append(A.PadIfNeeded(pad_to,pad_to, border_mode=cv2.BORDER_CONSTANT, value=0, position='top_left'))
    test_augs.append(A.Normalize(mean=(DATASET_MEAN,), std=(DATASET_STD,)))
    test_aug = A.Compose(test_augs)

    ds = ACDCDatasetAlbu(os.path.expanduser(dataset_root), transforms=test_aug, split='test')
    # Limit the number of samples considered in the test dataset. IF num_samples < len(ds). Otherwise use the full test set
    indeces = range(0, min(num_samples, len(ds))) 
    ds = Subset(ds, indeces)

    ious = []
    per_class_ious = []
    losses = []
    confusion_matrix = np.zeros((model.num_classes, model.num_classes))
    for idx, data in tqdm(enumerate(ds), total=min(num_samples, len(ds))):
        img, targets = data
        targets = torch.from_numpy(targets.astype(np.int64))

        input_ = utils.pad_to_next_multiple_of_32(img)
        
        input_ = torch.from_numpy(input_)
        input_ = input_.unsqueeze(dim=0)  # add minibatch dimension
        logits = model(input_)
        logits = logits[0, :, :img.shape[-2], :img.shape[-1]] # Remove padding and the minibatch dim
        logits = logits.detach().cpu()
        preds = torch.argmax(logits,dim=0)
        iou = torchmetrics.functional.jaccard_index(preds, targets, 
                                                    ignore_index=None, absent_score=1.0, num_classes=model.num_classes)
        ious.append(iou)
        per_class_ious.append(torchmetrics.functional.jaccard_index(preds, targets, 
                              absent_score=np.NaN, num_classes=model.num_classes, average='none'))
        confusion_matrix += torchmetrics.functional.confusion_matrix(preds, targets, num_classes=model.num_classes).numpy()
        preds = preds.numpy()
        
        file_id, slice_id, img_path, target_path = ds.dataset.samples[idx]
        original_img = ds.dataset.img_cache[file_id][... , slice_id] # acdc_utils.load_acdc_img(img_path)
        gt_mask = ds.dataset.gt_cache[file_id][... , slice_id]
        preds = preds[:original_img.shape[0], :original_img.shape[1]]

        logits = logits[:original_img.shape[0], :original_img.shape[1]]
        loss = model.loss(logits.unsqueeze(dim=0), targets.unsqueeze(dim=0))
        losses.append(loss)
        
        plot_title = img_path.replace(dataset_root+"/", "") + "\n" + target_path.replace(dataset_root+"/", "") + f"\nIoU: Mean = {iou:.3f}"+ \
        f" | RV = {per_class_ious[-1][1]:.3f} | MYO = {per_class_ious[-1][2]:.3f} | LV = {per_class_ious[-1][0]:.3f}"
       
        out_path = img_path.replace(".nii.gz", f"_{slice_id}.pdf")
        out_path = out_path.replace(dataset_root, PRED_FOLDER)
        out_path = os.path.join(output_folder, out_path)

        utils.plot_acdc_prediction(original_img, preds, gt_mask, plot_title, out_path)

    mean_iou =  np.mean(ious)
    std_iou = np.std(ious)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    per_class_ious_averaged_over_all_samples = np.nanmean(np.stack(per_class_ious), axis=0)
    np.save(os.path.join(output_folder,"per_class_ious"), np.stack(per_class_ious))
    print("=============================")
    print("Checkpoint", checkpoint_path)
    print(f"Evaluated {len(ious)} samples")
    print("Images padded to = ", pad_to)
    print("Mean IoU =",mean_iou)
    print("Stdev IoU =", std_iou)
    print("Mean Loss =",mean_loss)
    print("Stdev Loss =", std_loss)
    print("Per-class-IoUs", per_class_ious_averaged_over_all_samples)
    print("Mean of per-class-IoUs", np.mean(per_class_ious_averaged_over_all_samples))
    print("Confusion matrix: \n", confusion_matrix)
    print("=============================")

    if not os.path.exists(os.path.join(output_folder, PRED_FOLDER)):
        os.makedirs(os.path.join(output_folder, PRED_FOLDER), exist_ok=True)
    labels = ["Background", "Right ventrice (RV)", "Myocardium (MYO)", "Left ventrice (LV)"]
    utils.plot_iou_histograms(np.stack(per_class_ious), labels, os.path.join(output_folder, PRED_FOLDER))
    labels = ["BG", "RV", "MYO", "LV"]
    utils.plot_confmat(confusion_matrix, labels, os.path.join(output_folder, PRED_FOLDER))

    if merge_prediction_pdfs:
        utils.merge_pdfs(output_folder, PRED_FOLDER, out_file_name="ACDC-Predictions-Test.pdf", remove_files=remove_individual_pdfs)


    return mean_iou, std_iou, per_class_ious_averaged_over_all_samples, mean_loss, std_loss


def multi_ckpt_eval(ckpt_paths):
    for ckpt_path in ckpt_paths:
        assert os.path.exists(ckpt_path), ckpt_path
    results = []
    for ckpt_path in ckpt_paths:
        mean_iou, std_iou, per_class_ious, mean_loss, std_loss = generate_prediction_pdfs(
            ckpt_path,
            output_folder=os.path.dirname(os.path.dirname(ckpt_path)),
            merge_prediction_pdfs=True,
            remove_individual_pdfs=True
            )
        
        ckpt_path_parts = ckpt_path.split(os.path.sep)
        checkpoint = ckpt_path_parts[-1]
        version = ckpt_path_parts[-3]
        experiment = ckpt_path_parts[-4]
        if version != "version_0":
            ckpt_str = os.path.join(experiment, version, "...", checkpoint)
        else:
            ckpt_str = os.path.join(experiment, "...", checkpoint)
        ckpt_str = "`" + ckpt_str + "`"
        results.append([ckpt_str, mean_iou, std_iou, *per_class_ious])
    
    headers = ["Checkpoint", "Mean IoU", "Std IoU", "BG IoU", "RV", "MYO", "LV"]
    print(tabulate(results, headers, tablefmt="pipe"))


if __name__ == '__main__':
    ckpt_paths = [
        "artifacts-acdc/supervised/experiment_name/version_x/checkpoints/checkpoint.ckpt",
        # e.g.: "artifacts-acdc/supervised/ACDC/version_0/checkpoints/epoch=121-step=3171.ckpt"
        ]
    
    multi_ckpt_eval(ckpt_paths)

    # multi_resolution_eval(ckpt_paths[0])
 