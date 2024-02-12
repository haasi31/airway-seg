import nibabel as nib
import glob
import monai
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from multiprocessing import Pool
import argparse

def get_deform_transform(sigma_range=(8, 10), magnitude_range=(100, 150)):
    transform = monai.transforms.Compose([
        #monai.transforms.CropForegroundd(keys=('image', 'label', 'mask'), source_key='mask'),
        monai.transforms.Rand3DElasticd(
            prob=1.0,
            keys=('image', 'label', 'mask'),
            mode=('bilinear', 'nearest', 'nearest'),
            padding_mode='border',
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
            
            #rotate_range=(np.pi / 8,),
            #shear_range=(1.,),
            #translate_range=(5,),
            #scale_range=(0.1,)
        ),
    ])
    return transform


def get_crop_transform():
    return monai.transforms.CropForegroundd(keys=('image', 'label', 'mask'), source_key='mask', margin=5, allow_smaller=True)


def deform(kwargs):
    volume_file, label_file, mask_file, out_dir, args = kwargs.values()
    base_name = '_'.join(volume_file.split('/')[-1].split('_')[:-1])
    name_extension = '' #f'_s{hyper_parameter["sigma_range"]}_m{hyper_parameter["magnitude_range"]}'
    base_name += name_extension
    
    # if os.path.exists(f'{out_dir}/images/{base_name}_0_volume.nii.gz'):
    #     print(f'{out_dir}/images/{base_name}_0_volume.nii.gz already exists, skipping')
    #     return
    
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
        # nifti = nib.Nifti1Image(data['image'][0].numpy().astype(np.int16), affine)
        # nib.save(nifti, f'{out_dir}/images/{base_name}_0_volume.nii.gz')
        nifti = nib.Nifti1Image(data['label'][0].numpy().astype(np.uint8), affine)
        nib.save(nifti, f'{out_dir}/labels/{base_name}_0_label.nii.gz')
        # nifti = nib.Nifti1Image(data['mask'][0].numpy().astype(np.uint8), affine)
        # nib.save(nifti, f'{out_dir}/masks/{base_name}_0_mask.nii.gz')
    
    if args['deformation']:
        for i in range(args['num_deformations']):
            deformed_data = get_deform_transform()(data)
            
            nifti = nib.Nifti1Image(deformed_data['image'][0].numpy().astype(np.int16), affine)
            nib.save(nifti, f'{out_dir}/images/{base_name}_{i+1}_volume.nii.gz')
            nifti = nib.Nifti1Image(deformed_data['label'][0].numpy().astype(np.uint8), affine)
            nib.save(nifti, f'{out_dir}/labels/{base_name}_{i+1}_label.nii.gz')
            nifti = nib.Nifti1Image(deformed_data['mask'][0].numpy().astype(np.uint8), affine)
            nib.save(nifti, f'{out_dir}/masks/{base_name}_{i+1}_mask.nii.gz')   


def main():
    # base_dir = '/home/ahaas/data/2_manual_adapted_data/20_samples_per_7_base'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='/home/ahaas/data/3_deformed_data')
    parser.add_argument('--crop_foreground', action='store_true')
    parser.add_argument('--save_original', action='store_true')
    parser.add_argument('--deformation', action='store_true')
    parser.add_argument('--num_deformations', type=int, default=2)
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads to use for multiprocessing')
    
    
    args = parser.parse_args()
    input_dir = args.input_dir # '/home/ahaas/data/3_deformed_data/20_samples_per_7_base'
    out_dir = args.out_dir # '/home/ahaas/data/3_deformed_data/hyperparameter_search_3'
    crop_foreground = args.crop_foreground
    save_original = args.save_original
    deformation = args.deformation
    num_deformations = args.num_deformations
    
    volume_files = sorted(glob.glob(f'{input_dir}/images/*'))
    label_files = sorted(glob.glob(f'{input_dir}/labels/*_airways.nii.gz'))
    mask_files = sorted(glob.glob(f'{input_dir}/masks/*'))
    assert len(volume_files) == len(label_files) == len(mask_files)
    
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/labels', exist_ok=True)
    os.makedirs(f'{out_dir}/masks', exist_ok=True)
    os.makedirs(f'{out_dir}/visualization', exist_ok=True)
    data_files = [(volume_file, label_file, mask_file, out_file) 
                  for volume_file, label_file, mask_file, out_file 
                  in zip(volume_files, label_files, mask_files, [out_dir,]*len(volume_files))]
    

    if crop_foreground:
        crop_transform = get_crop_transform()
    if deformation:
        deform_transform = get_deform_transform()
     
    data = []
    for volume_file, label_file, mask_file, out_dir in data_files:
        data.append({
            'volume_file': volume_file,
            'label_file': label_file,
            'mask_file': mask_file,
            'out_dir': out_dir,
            'args': {'crop_foreground': crop_foreground,
                     'save_original': save_original, 
                     'deformation': deformation, 
                     'num_deformations': num_deformations}
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