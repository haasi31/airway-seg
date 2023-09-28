import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, binary_erosion
import glob
from tqdm import tqdm
import os
import shutil
import yaml
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


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
    
    
def adapt_airways_and_vessels(args):
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    airway_dir, vessel_dir, out_dir, orig_dir = args
    hash = airway_dir.split('/')[-1]
    with open(f'{airway_dir}/config.yml', "r") as filepath:
        config = yaml.safe_load(filepath)
    scan = config['Forest']['segment_root_path'][-13:-10]
    
    airway = nib.load(airway_dir + '/volume.nii.gz').get_fdata().astype(np.float32)
    airway_mask = airway > 0 #biggest mask

    lumen = nib.load(airway_dir + '/lumen.nii.gz').get_fdata().astype(np.float32)
    lumen_mask = lumen > 0 #small mask

    #smallest mask
    label_mask = nib.load(airway_dir + '/airways.nii.gz').get_fdata().astype(np.uint8) >= 1

    #volume = nib.load(airway_dir + '/airway_wall.nii.gz').get_fdata().astype(np.float32)

    vessels = nib.load(vessel_dir + '/volume.nii.gz').get_fdata().astype(np.float32)
    vessels_mask = vessels > 0

    lung_mask = nib.load(airway_dir + '/lung.nii.gz').get_fdata().astype(np.uint8)
    lung_mask = lung_mask >= 1

    combined_mask = np.logical_or(airway_mask, vessels_mask) #combined_volume > 0
    #parenchyma_mask = np.logical_and(lung_mask, combined_mask == False)
    #blend_mask = binary_erosion(combined_mask)
    
    #HU mapping
    airway_mapping = lambda x: 220*(x+0.2)**2-915
    mapped_airway = airway_mapping(airway[airway_mask])
    lumen_mapping = lambda x: 205*(x-1.35)**2-1050
    mapped_lumen = lumen_mapping(lumen[label_mask])

    vessel_mapping = lambda x: 450 * (x+0.4)**2.5 - 980
    mapped_vessels = vessel_mapping(vessels[vessels_mask])
    vessel_noise = np.random.normal(-10, 5, np.sum(vessels_mask))
    mapped_vessels += vessel_noise

    adapted_volume = np.zeros_like(lumen)
    adapted_volume[vessels_mask] = mapped_vessels
    adapted_volume[airway_mask] = mapped_airway
    adapted_volume[label_mask] = mapped_lumen
    
    del mapped_airway
    del mapped_lumen
    del mapped_vessels
    
    noise = np.random.uniform(0, 1, lumen.shape)   
    noise_smoothed = gaussian_filter(noise, sigma=(0.8, 0.8, 0.9))
    
    # histogram equalization
    min, max = noise_smoothed.min(), noise_smoothed.max()
    noise_smoothed = (noise_smoothed-min) / (max-min) #normalize to [0,1]

    bins = 1000 #max-min
    hist, _ = np.histogram(noise_smoothed, bins=bins)
    cdf1 = hist.cumsum().astype(np.float64)
    cdf1 /= cdf1.max() #normalize to [0,1]

    x = np.linspace(0, 1, bins) 
    mu = -2.5
    sig = 0.3
    target_cdf = 1/(x*sig*np.sqrt(2*np.pi)+1e-5)*np.exp(-(np.log(x+1e-5)-mu)**2/(2*sig**2))
    target_cdf = target_cdf.cumsum()
    target_cdf /= target_cdf.max() #normalize to [0,1]

    min, max = 0, 1
    mapping_function = np.interp(cdf1, target_cdf, np.arange(min, max, (max-min)/bins))

    noise_equalized = np.interp(noise_smoothed.ravel(), np.arange(0, 1, 1/bins), mapping_function)
    noise_equalized = noise_equalized.reshape(noise_smoothed.shape)
    
    noise_equalized_scaled = noise_equalized * 1250 - 1000
    
    del noise_smoothed
    del noise_equalized
    orig_img = nib.load(f'{orig_dir}/images/ATM_{scan}_0000.nii.gz')
    orig_vol = orig_img.get_fdata()
    affine = orig_img.affine
    del orig_img
    
    #blend_factor = 0.9
    final_volume = noise_equalized_scaled
    #final_volume = np.ones_like(adapted_volume) * (-950)
    # final_volume = np.where(lung_mask, final_volume, orig_vol) # add background
    # final_volume = np.where(parenchyma_mask, noise_equalized_scaled, final_volume) # add parenchyma
    final_volume = np.where(combined_mask, adapted_volume, final_volume) # add airways and vessels
    #final_volume = np.where(blend_mask, adapted_volume*blend_factor+noise_equalized_scaled*(1-blend_factor), final_volume) # blend edge of airways and vessels
    final_volume = np.where(lung_mask, final_volume, orig_vol) # add background
    
    # saving
    #histogram
    voxels1 = final_volume[lung_mask].ravel()
    voxels2 = orig_vol[lung_mask].ravel()
    plt.figure()
    plt.hist(voxels2, bins=200, label='orig_vol', histtype='step')
    plt.hist(voxels1, bins=200, label='adapted_vol', histtype='step')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{out_dir}/hist/{hash}_histogram.png')
    plt.close()
    
    # volume
    nifti = nib.Nifti1Image(final_volume.astype(np.int16), affine)
    nib.save(nifti, f"{out_dir}/images/{hash}_volume.nii.gz")

    # vessels
    vessels_seg = vessels > 0.25
    vessels_seg = np.where(lumen_mask, 0, vessels_seg)
    #vessels_seg[lung_mask == False] = 0
    vessels_seg = np.logical_and(vessels_seg, lung_mask)
    nifti = nib.Nifti1Image(vessels_seg.astype(np.uint8), affine)
    nib.save(nifti, f"{out_dir}/labels/{hash}_vessels.nii.gz")
    
    # airways
    #label_mask[lung_mask == False] = 0
    label_mask = np.logical_and(label_mask, lung_mask)
    nifti = nib.Nifti1Image(label_mask.astype(np.uint8), affine)
    nib.save(nifti, f"{out_dir}/labels/{hash}_airways.nii.gz")
    
    
def create_dataset():
    dirs = glob.glob(os.path.abspath('vessel_graph_generation/datasets/lobes/*/'))
    for root_dir in tqdm(dirs):
        shutil.copyfile(root_dir + '/adapted_vol.nii.gz', )
    
    
def main():
    airway_dirs = sorted(glob.glob(os.path.abspath('vessel_graph_generation/datasets/airways/*')))
    vessel_dirs = sorted(glob.glob(os.path.abspath('vessel_graph_generation/datasets/blood_vessels/*')))
    orig_dir = '/home/shared/Data/ATM22/train/'
    out_dir = 'vessel_graph_generation/datasets/dataset_4.2_overfit'
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/labels', exist_ok=True)
    os.makedirs(f'{out_dir}/hist', exist_ok=True)
    args = []
    for airway_dir, vessel_dir in zip(airway_dirs, vessel_dirs):
        args.append((airway_dir, vessel_dir, out_dir, orig_dir))
    with Pool(10) as p, tqdm(total=len(args)) as pbar:
        for _ in p.imap_unordered(adapt_airways_and_vessels, args):
            pbar.update() #adapt_airways_and_vessels(airway_dir, vessel_dir, out_dir, orig_dir)
          

if __name__ == '__main__':
    main()
