import os
import glob
import re
import numpy as np
import matplotlib
from tqdm import tqdm

import nibabel as nib

acdc_cmap=np.array(
   [[0.        , 0.        , 0.        , 0.],
    [0.89411765, 0.10196078, 0.10980392, 1.],
    [0.30196078, 0.68627451, 0.29019608, 1.],
    [1.        , 0.49803922, 0.        , 1.]])
acdc_cmap = matplotlib.colors.ListedColormap(acdc_cmap)


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header

def load_acdc_img(path, dtype=np.uint16):
    img, _, _ = load_nii(path)
    img = img.astype(dtype)
    if '_4d.nii' in path:
        img = np.reshape(img,(img.shape[0], img.shape[1], -1))
    return img


def construct_samples_list(root, include_unlabelled=False):
    """
    E.g.
        file_id    slice_id  image_filename             label_filename
    ---------  ----------  -------------------------  ----------------------------
            0           0  patient001_frame01.nii.gz  patient001_frame01_gt.nii.gz
          ...
            0           9  patient001_frame01.nii.gz  patient001_frame01_gt.nii.gz
            1           0  patient001_frame12.nii.gz  patient001_frame12_gt.nii.gz
            1           1  patient001_frame12.nii.gz  patient001_frame12_gt.nii.gz
          ...
          199           7  patient100_frame13.nii.gz  patient100_frame13_gt.nii.gz

    Args:
        root (str): dataset root folder

    Returns:
        list of tuples (int, int, str, str): List of tuples, each with the following elements: file_id, slice_id, image_filename, label_filename
    """
    samples = []
    img_cache = {}
    gt_cache = {}
    nii_gz_paths = sorted(glob.glob(os.path.join(root,'**', '*.nii.gz'), recursive=True))
    nii_gz_paths = [path_ for path_ in nii_gz_paths if '_gt.nii' not in path_ ]
    if not include_unlabelled: # drop unlabelled files
        nii_gz_paths = [path_ for path_ in nii_gz_paths if '_4d.nii' not in path_]
    for file_id, nii_gz_path in tqdm(enumerate(nii_gz_paths), total=len(nii_gz_paths), desc="Constucting samples list from files"):
        img_data = load_acdc_img(nii_gz_path)
        num_slices = img_data.shape[2]
        img_path = nii_gz_path
        if '_4d.nii' in nii_gz_path:
            # Unlabelled sample
           samples.extend([(file_id, i, img_path, "") for i in range(num_slices)])
        else:
            # Labeled sample
            gt_path = nii_gz_path.replace('.nii', '_gt.nii')
            samples.extend([(file_id, i, img_path, gt_path) for i in range(num_slices)])
            gt_data = load_acdc_img(gt_path, dtype = np.uint8)
            gt_cache[file_id] = gt_data
        img_cache[file_id] = img_data
    
    print(f"Found {len(samples)} samples.")
    return samples, img_cache, gt_cache


def split_samples_list(samples, split_patient_ids):
    split_samples = []
    for sample in samples:
        img_filename = os.path.basename(sample[2]) # e.g. patient001_frame01.nii.gz
        patient_id = int(img_filename[7:10]) ## ACDC dataset specific
        if patient_id in split_patient_ids:
            split_samples.append(sample)
    return split_samples


def get_dataset_mean_std(root: str):
    """
    Args:
        root (str): dataset root folder

    Returns:
        (float, float): mean, std
    """
    all_pixels = []
    nii_gz_paths = sorted(glob.glob(os.path.join(root,'**', '*.nii.gz'), recursive=True))
    nii_gz_paths = [path_ for path_ in nii_gz_paths 
                    if '_gt.nii' not in path_ 
                    and '_4d.nii' not in path_]
    for nii_gz_path in tqdm(nii_gz_paths, desc="Loading samples for mean, std computation"):
        data, _, _ = load_nii(nii_gz_path)
        all_pixels.append(data.flatten())
    all_pixels = np.concatenate(all_pixels)
    return all_pixels.mean(), all_pixels.std()