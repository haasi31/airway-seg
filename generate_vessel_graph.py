import argparse
import copy
import csv
from vessel_graph_generation.forest import Forest
from vessel_graph_generation.greenhouse import Greenhouse
from vessel_graph_generation.utilities import prepare_output_dir, read_config
import vessel_graph_generation.tree2img as tree2img
import math
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import yaml
from multiprocessing import cpu_count
import concurrent.futures
import warnings


def main(cfg):
    dir = prepare_output_dir(cfg['output'])
    roots = read_config(cfg['Forest']['segment_root_path'])
    volume, lobes, segmentation, airway_wall = [], [], [], []
    for i in tqdm(range(1, 6)):
        # manipulate config for each lobe
        config = copy.deepcopy(cfg)
        config['Greenhouse']['SimulationSpace']['oxygen_sample_geometry_path'] = \
            cfg['Greenhouse']['SimulationSpace']['oxygen_sample_geometry_path'].format(i)
        # config['Forest']['bronchi_pos'] = [cfg['Forest']['bronchi_pos'][i-1]]
        # config['Forest']['bronchi_dir'] = [cfg['Forest']['bronchi_dir'][i-1]]
        config['Forest']['roots'] = roots[f'lobe{i}']

        # Initialize greenhouse
        greenhouse = Greenhouse(config['Greenhouse'])
        # Prepare output directory
        out_dir = dir + f'/{i}/'
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'config.yml'), 'w') as f:
            yaml.dump(config, f)

        # Initialize forest
        arterial_forest = Forest(config['Forest'], greenhouse.d, greenhouse.r, greenhouse.simspace)
        venous_forest = None

        greenhouse.set_forests(arterial_forest, venous_forest)

        # Grow vessel network
        greenhouse.develop_forest()
        if config["output"]["save_stats"]:
            greenhouse.save_stats(out_dir)

        # volume_dimension = [int(d) for d in greenhouse.simspace.shape*config['output']['image_scale_factor']]
        volume_dimension = [math.ceil(d) for d in greenhouse.simspace.shape*greenhouse.simspace.geometry_size]

        art_edges = [{
            "node1": current_node.position,
            "node2": current_node.get_proximal_node().position,
            "radius": current_node.radius
        } for tree in arterial_forest.get_trees() for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False)]

        # Save vessel graph as csv file
        if config['output']['save_trees']:
            name = out_dir.split("/")[-2]
            filepath = os.path.join(out_dir, name+'.csv')
            with open(filepath, 'w+') as file:
                writer = csv.writer(file)
                writer.writerow(["node1", "node2", "radius"])
                edges = art_edges
                for row in edges:
                    writer.writerow([row["node1"], row["node2"], row["radius"]])

        radius_list=[]
        if config["output"].get("save_3D_volumes"):
            art_mat, _, lobe, seg, wall = tree2img.voxelize_forest(art_edges, volume_dimension, radius_list, greenhouse.simspace, config['output'])
            if config["output"]["save_3D_volumes"] == "npy":
                np.save(f'{out_dir}/art_img_gray.npy', art_mat_gray)
                np.save(f'{out_dir}/lob.npy', lobe)
                np.save(f'{out_dir}/seg.npy', seg)
            elif config["output"]["save_3D_volumes"] == "nifti":
                nifti = nib.Nifti1Image(art_mat, np.eye(4))
                nib.save(nifti, f"{out_dir}/gray.nii.gz")
                nifti = nib.Nifti1Image(lobe, np.eye(4))
                nib.save(nifti, f"{out_dir}/lobe.nii.gz")
                nifti = nib.Nifti1Image(seg, np.eye(4))
                nib.save(nifti, f"{out_dir}/seg.nii.gz")
            volume.append(art_mat)
            lobes.append(lobe)
            segmentation.append(seg)
            airway_wall.append(wall)

        if config["output"]["save_2D_image"]:
            radius_list=[]
            image_res = [*volume_dimension]
            del image_res[config["output"]["proj_axis"]]
            sim_shape = [*greenhouse.simspace.shape]
            del sim_shape[config["output"]["proj_axis"]]
            art_mat,_ = tree2img.rasterize_forest(art_edges, image_res, MIP_axis=config["output"]["proj_axis"], radius_list=radius_list, sim_shape=sim_shape)
            art_mat_gray = art_mat.astype(np.uint8)
            tree2img.save_2d_img(art_mat_gray, out_dir, "art_ven_img_gray")

        if config["output"]["save_stats"]:
            tree2img.plot_vessel_radii(out_dir, radius_list)

    with open(os.path.join(dir, 'config.yml'), 'w') as f:
        yaml.dump(cfg, f)
    # merge 5 lobes
    for i in range(1, 6):
        lobes[i-1] *= i
        segmentation[i-1] *= i
        airway_wall[i-1] *= i
    volume = np.max(np.array(volume), axis=0)
    lung = np.max(np.array(lobes), axis=0)
    airways = np.max(np.array(segmentation), axis=0)
    airway_wall = np.max(np.array(airway_wall), axis=0)

    # np.save(f'{dir}/volume.npy', volume)
    # np.save(f'{dir}/lung.npy', lung)
    # np.save(f'{dir}/airways.npy', airways)
    # np.save(f'{dir}/airway_wall.npy', airway_wall)

    nifti = nib.Nifti1Image(volume, np.eye(4))
    nib.save(nifti, f"{dir}/volume.nii.gz")
    nifti = nib.Nifti1Image(lung, np.eye(4))
    nib.save(nifti, f"{dir}/lung.nii.gz")
    nifti = nib.Nifti1Image(airways, np.eye(4))
    nib.save(nifti, f"{dir}/airways.nii.gz")
    nifti = nib.Nifti1Image(airway_wall, np.eye(4))
    nib.save(nifti, f"{dir}/airway_wall.nii.gz")


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--threads', help="Number of parallel threads. By default all available threads but one are used.", type=int, default=-1)
    args = parser.parse_args()

    if args.debug:
        warnings.filterwarnings('error')

    # Read config file
    assert os.path.isfile(args.config_file), f"Error: Your provided config path {args.config_file} does not exist!"
    config = read_config(args.config_file)

    assert config['output'].get('save_3D_volumes') in [None, 'npy', 'nifti'], f"Your provided option {config['output'].get('save_3D_volumes')} for 'save_3D_volumes' does not exist. Choose one of 'null', 'npy' or 'nifti'."

    if args.threads == -1:
        # If no argument is provided, use all available threads but one
        cpus = cpu_count()
        threads = min(cpus-1 if cpus>1 else 1,args.num_samples)
    else:
        threads=args.threads
    if threads>1:
        # Multi processing
        with tqdm(total=args.num_samples, desc="Generating vessel graphs...") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                future_dict = {executor.submit(main, config): i for i in range(args.num_samples)}
                for future in concurrent.futures.as_completed(future_dict):
                    i = future_dict[future]
                    pbar.update(1)
    else:
        # Single processing
        for i in tqdm(range(args.num_samples), desc="Generating vessel graphs..."):
            main(config)
