import csv
import yaml
import numpy as np
import nibabel as nib
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
import os

from vessel_graph_generation import tree2img


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--threads', help="Number of parallel threads. By default all available threads but one are used.", type=int, default=-1)
    parser.add_argument('--input_dir', help="Path to the directory containing the simulation directories.", type=str, required=True)
    parser.add_argument('--scans', nargs='+', type=str, help="List of ATM scans to process", default=None)
    parser.add_argument('--min_radius', type=float, help='Minimum radius of the airways to be considered', default=0)
    args = parser.parse_args()
    
    if args.threads == -1:
        # If no argument is provided, use all available threads but one
        cpus = cpu_count()
        threads = min(cpus-1 if cpus>1 else 1,args.num_samples)
    else:
        threads=args.threads
    input_dir = args.input_dir
    mode = 'airways' # 'vessels', '*'
    if args.scans is not None:
        sim_dirs = []
        for scan in args.scans:
            sim_dirs += glob.glob(f'{input_dir}/ATM_{scan}_*/{mode}/')
    else:
        sim_dirs = glob.glob(f'{input_dir}/*/{mode}/')
    args_list = [{'sim_path': sim_path, 'min_radius': args.min_radius} for sim_path in sim_dirs]

    if threads > 1:
        with Pool(threads, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p, \
            tqdm(total=len(args_list), position=0, desc="Voxelization...", leave=True) as pbar:
            
            for _ in p.imap_unordered(csv2nifti, args_list):
                pbar.update()
    else: 
        for args in tqdm(args_list):
            csv2nifti(args)


def csv2nifti(args):
    sim_path, min_radius = args['sim_path'], args['min_radius']
    volume, segmentation, lumen, airway_wall, lobes = [], [], [], [], []
    
    with open(f'{sim_path}/lobe_1/config.yml', 'r') as file:
        config = yaml.safe_load(file)
    label_name = f'{config["mode"]}_label.nii.gz' if min_radius == 0 else f'{config["mode"]}_label_minradius{min_radius}.nii.gz'
    if os.path.exists(f"{sim_path}/{label_name}"):
        print(f"Skipping {sim_path} as {label_name} already exists.")
        return

    for lobe_idx in range(1, 6):
        with open(f'{sim_path}/lobe_{lobe_idx}/graph.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)
            edges = []
            for row in reader:
                edges.append({
                    'node1': np.fromstring(row[0][1:-1], dtype=float, sep=' '),
                    'node2': np.fromstring(row[1][1:-1], dtype=float, sep=' '),
                    'radius': float(row[2])})
        with open(f'{sim_path}/lobe_{lobe_idx}/graph_stats.yml', 'r') as file:
            stats = yaml.safe_load(file)
            volume_dimension = stats['volume_dimension']
            radius_list = stats['radius_list']
            geometry_path = stats['geometry']
            geometry = np.load(geometry_path)
        with open(f'{sim_path}/lobe_{lobe_idx}/config.yml', 'r') as file:
            config = yaml.safe_load(file)
            
            
                
        out_dict = tree2img.voxelize_forest(edges, volume_dimension, radius_list, geometry, config, min_radius=min_radius)
        mat, seg = out_dict['img'], out_dict['seg']
        volume.append(mat)
        segmentation.append(seg)
        if config['mode'] == 'airways':
            lumen_, wall, lobe = out_dict['lumen_img'], out_dict['wall'], out_dict['lobe']
            lumen.append(lumen_)
            #airway_wall.append(wall)
            lobes.append(lobe)
            
    volume = np.max(np.array(volume), axis=0)
    segmentation = np.max(np.array(segmentation), axis=0)
    if config['mode'] == 'airways':
        lumen = np.max(np.array(lumen), axis=0)
        #airway_wall = np.max(np.array(airway_wall), axis=0)
        lung = np.max(np.array(lobes), axis=0)
    
    scan = config['ATM_scan']
    affine = nib.load(f'/home/shared/Data/ATM22/train/images/ATM_{scan}_0000.nii.gz').affine

    #nib.save(nib.Nifti1Image(volume, affine), f"{sim_path}/volume.nii.gz")
    #label_name = f'{config["mode"]}_label.nii.gz' if min_radius == 0 else f'{config["mode"]}_label_minradius{min_radius}.nii.gz'
    nib.save(nib.Nifti1Image(segmentation, affine), f"{sim_path}/{label_name}")
    #if config['mode'] == 'airways':
    #    nib.save(nib.Nifti1Image(lumen, affine), f"{sim_path}/lumen.nii.gz")
        #nib.save(nib.Nifti1Image(airway_wall, affine), f"{sim_path}/airway_wall.nii.gz")
    #    nib.save( nib.Nifti1Image(lung, affine), f"{sim_path}/lung.nii.gz")


if __name__ == '__main__':
    main()
