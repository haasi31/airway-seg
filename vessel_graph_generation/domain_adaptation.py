import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import glob
from tqdm import tqdm
import os
import shutil


def adapt(root_dir, out_dir, scan):
    orig_img = nib.load(f'/home/shared/Data/ATM22/train/images/ATM_{scan}_0000.nii.gz')
    orig_vol = orig_img.get_fdata()
    orig_seg = nib.load(f'/home/shared/Data/ATM22/train/labels/ATM_{scan}_0000.nii.gz').get_fdata()
    affine = orig_img.affine
    
    vol = np.load(root_dir + '/volume.npy').astype(np.float32)
    vol = vol / 255

    lung_mask = np.load(root_dir + '/lung.npy').astype(np.uint8)
    lung_mask = lung_mask >= 1

    wall_mask = np.load(root_dir + '/airway_wall.npy').astype(np.uint8)
    wall_mask = wall_mask >= 1

    airway_mask = np.load(root_dir + '/airways.npy').astype(np.uint8)
    airway_mask = airway_mask >= 1
    
    noise = np.random.normal(0, 0.5, vol.shape)
    noise = gaussian_filter(noise, sigma=2)
    
    adapted_vol = lung_mask.astype(np.uint8) * 0.9
    adapted_vol[wall_mask] = vol[wall_mask] / 10
    adapted_vol[airway_mask] = vol[airway_mask]
    adapted_vol = gaussian_filter(adapted_vol, sigma=0.8)
    adapted_vol += noise
    adapted_vol *= -950
    adapted_vol[lung_mask == False] = orig_vol[lung_mask == False]
    adapted_vol = adapted_vol.astype(np.float32)
    nifti = nib.Nifti1Image(adapted_vol, affine) # np.eye(4))
    nib.save(nifti, f"{out_dir}/images/{root_dir.split('/')[-1]}_adapted_vol.nii.gz")
    
    main_bronchi = np.where(np.bitwise_and(lung_mask == False, orig_seg >= 1), True, False)
    adapted_airway = airway_mask.astype(np.uint8)
    adapted_airway[main_bronchi] = 1
    nifti = nib.Nifti1Image(adapted_airway, affine) #np.eye(4))
    nib.save(nifti, f"{out_dir}/labels/{root_dir.split('/')[-1]}_adapted_label.nii.gz")
    
    
def create_dataset():
    dirs = glob.glob(os.path.abspath('vessel_graph_generation/datasets/lobes/*/'))
    for root_dir in tqdm(dirs):
        shutil.copyfile(root_dir + '/adapted_vol.nii.gz', )
    
    
def main():
    dirs = sorted(glob.glob(os.path.abspath('vessel_graph_generation/datasets/lobes/20230810*')))
    out_dir = 'vessel_graph_generation/datasets/dataset_1'
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/labels', exist_ok=True)
    for root_dir in tqdm(dirs):
        adapt(root_dir, out_dir, '257')
        #shutil.copyfile(root_dir + '/airways.nii.gz', f"{out_dir}/labels/{root_dir.split('/')[-1]}_label.nii.gz")
        # label = nib.load(root_dir).get_fdata()
        # label = np.where(label >= 1, 1, 0).astype(np.uint8)
        # nifti = nib.Nifti1Image(label, np.eye(4))
        # nib.save(nifti, root_dir)
          

if __name__ == '__main__':
    main()
