import nibabel as nib
import glob
import monai
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from multiprocessing import Pool
import argparse


def get_crop_transform():
    return monai.transforms.CropForegroundd(keys=('image', 'label', 'mask'), source_key='mask', margin=5, allow_smaller=True)


def deform(kwargs):
    volume_file, label_file, mask_file, out_dir, args = kwargs.values()
    base_name = '_'.join(volume_file.split('/')[-1].split('_')[:-1])
    
    affine = nib.load(volume_file).affine
    volume = nib.load(volume_file).get_fdata()
    label = nib.load(label_file).get_fdata()
    mask = nib.load(mask_file).get_fdata()
    
    data = {
        'image': np.expand_dims(volume, axis=0),
        'label': np.expand_dims(label, axis=0),
        'mask': np.expand_dims(mask, axis=0),
    }
    
    if args['crop_foreground']:
        data = get_crop_transform()(data)

    if args['save_original']:
        nifti = nib.Nifti1Image(data['image'][0].numpy().astype(np.int16), affine)
        nib.save(nifti, f'{out_dir}/images/{base_name}_0_volume.nii.gz')
        nifti = nib.Nifti1Image(data['label'][0].numpy().astype(np.uint8), affine)
        nib.save(nifti, f'{out_dir}/labels/{base_name}_0_label.nii.gz')
        nifti = nib.Nifti1Image(data['mask'][0].numpy().astype(np.uint8), affine)
        nib.save(nifti, f'{out_dir}/masks/{base_name}_0_mask.nii.gz')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='/home/ahaas/data/3_deformed_data')
    parser.add_argument('--crop_foreground', action='store_true')
    parser.add_argument('--save_original', action='store_true')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads to use for multiprocessing')
    
    args = parser.parse_args()
    input_dir = args.input_dir
    out_dir = args.out_dir
    crop_foreground = args.crop_foreground
    save_original = args.save_original
    
    volume_files = sorted(glob.glob(f'{input_dir}/images/*'))
    label_files = sorted(glob.glob(f'{input_dir}/labels/*_airways.nii.gz'))
    mask_files = sorted(glob.glob(f'{input_dir}/masks/*'))
    assert len(volume_files) == len(label_files) == len(mask_files)
    
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/labels', exist_ok=True)
    os.makedirs(f'{out_dir}/masks', exist_ok=True)
    data_files = [(volume_file, label_file, mask_file, out_file) 
                  for volume_file, label_file, mask_file, out_file 
                  in zip(volume_files, label_files, mask_files, [out_dir,]*len(volume_files))]
     
    data = []
    for volume_file, label_file, mask_file, out_dir in data_files:
        data.append({
            'volume_file': volume_file,
            'label_file': label_file,
            'mask_file': mask_file,
            'out_dir': out_dir,
            'args': {'crop_foreground': crop_foreground,
                     'save_original': save_original}
        })
    if args.threads > 1:
        with Pool(args.threads) as p, tqdm.tqdm(total=len(data_files)) as pbar:
            for _ in p.imap_unordered(deform, data):
                pbar.update()
    else:
        for d in tqdm.tqdm(data):
            deform(d)


if __name__ == '__main__':
    main()