import os
from PIL import Image
from tqdm import tqdm
from data.acdc_utils import construct_samples_list, split_samples_list


NUM_PATIENTS = 100
TEST_PATIENTS = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
VAL_PATIENTS = [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
TRAIN_PATIENTS = [
    id for id in range(1, NUM_PATIENTS + 1) if id not in TEST_PATIENTS and id not in VAL_PATIENTS
]


def acdc2mmseg(
    root="~/data/acdc",
    out_path="data/acdc",
    file_extension=".tiff",
    train_patients=TRAIN_PATIENTS,
    val_patients=VAL_PATIENTS,
    test_patients=TEST_PATIENTS,
):
    """_summary_
        .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    Args:
        root (_type_): _description_
        out_path (str, optional): _description_. Defaults to "data/acdc".
        train_patients (_type_, optional): _description_. Defaults to TRAIN_PATIENTS.
        val_patients (_type_, optional): _description_. Defaults to VAL_PATIENTS.
        test_patients (_type_, optional): _description_. Defaults to TEST_PATIENTS.
    """
    split_patient_ids = {"train": train_patients, "val": val_patients, "test": test_patients}
    all_samples, img_cache, gt_cache = construct_samples_list(os.path.expanduser(root))

    for split in split_patient_ids.keys():
        split_samples = split_samples_list(all_samples, split_patient_ids[split])
        out_img_folder = os.path.join(out_path, "img_dir", split)
        out_ann_folder = os.path.join(out_path, "ann_dir", split)
        os.makedirs(out_img_folder, exist_ok=True)
        os.makedirs(out_ann_folder, exist_ok=True)

        for file_id, slice_id, img_path, target_path in tqdm(split_samples, desc=f"Exporting {split} samples"):
            sample = img_cache[file_id][... , slice_id]
            target = gt_cache[file_id][... , slice_id]

            sample_out_filename = os.path.basename(img_path).split('.')[0] + f"_{slice_id}" 
            sample_out_path = os.path.join(out_img_folder, sample_out_filename + file_extension)
            target_out_path = os.path.join(out_ann_folder, sample_out_filename + "_gt" + file_extension)
            
            Image.fromarray(sample).save(sample_out_path)
            Image.fromarray(target).save(target_out_path)

if __name__ == '__main__':
    acdc2mmseg()