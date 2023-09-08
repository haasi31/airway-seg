from typing import Literal, Tuple, Sequence
from random import random

import numpy as np
from numpy import floor

import math
import itertools
from PIL import Image
from matplotlib import pyplot as plt, collections, cm
from scipy.ndimage import zoom


def rasterize_forest(forest: list[dict],
                     image_resolution: Sequence[float],
                     MIP_axis: int=2,
                     radius_list: list=None, 
                     min_radius: float=0,
                     max_radius: float=1,
                     max_dropout_prob=0,
                     blackdict: dict[str, bool]=None,
                     colorize: str=None,
                     sim_shape: tuple=None) -> Tuple[np.ndarray, dict[str, bool]]:
    """
    Converts the given 3D forest into a 2D (grayscale) image.
    Antialiased drawing of the tree edges is performed by matplotlib.

    Parameters:
    -----------
        - forest: list of edges. An edge is a dictionary with 'node1' position 'node2' position and 'radius'.
        - image_resolution: Dimensions of the final 2D image
        - MIP_axis: Axis along which to take the maximum intensity projection. Default is the z dimension
        - radius_list: A list to collect all edge radii. Default is None
        - min_radius: All edges with radius smaller than this will not be included in the grayscale image
        - max_radius: All edges with radius larger than this will not be included in the grayscale image
        - max_dropout_prob: Maximum probablity with which an edge and its decendents are dropped. 
                            The probabily is sampled from :math:`p = P**10, P ~ Uniform(0,max_dropout_prob)`
        - blackdict: A dictionary containing all parent nodes that were removed in the paired image.
                    All edges from these nodes will be dropped. If a dictionary is provided, no other edges will be removed.
        - colorize: If set, an RGB image will be returned where the vessels are colorized depending on their radius.
                    - 'continuos': Continously colorize the vessels using the 'plasma' colormap.
                    - 'discrete': Divide the vessel radii into three intervals.

    Returns:
    -------
        - 2D image of all vessels. RGB if colorized, else grayscale
        - backlist dictionary containing all parent nodes that were dropped 
    """
    axes = [a for a in [0,1,2] if a!=MIP_axis]
    if radius_list is None:
        radius_list=[]
    no_pixels_x, no_pixels_y = image_resolution
    scale_factor = max(no_pixels_x, no_pixels_y)
    dpi = 100
    x_inch = no_pixels_x / dpi
    y_inch = no_pixels_y / dpi
    figure = plt.figure(figsize=(y_inch, x_inch))
    figure.patch.set_facecolor('black')
    ax = plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    ax.invert_yaxis()
    edges = []
    radii = []
    if blackdict is None:
        blackdict = dict()
        p = random()**10 * max_dropout_prob
    else:
        p=0
    for edge in forest:
        radius = float(edge["radius"])
        if radius<min_radius or radius>max_radius:
            continue
        radius *= 1.3

        if isinstance(edge["node1"], np.ndarray) or isinstance(edge["node1"], list):
            current_node = tuple(edge["node1"])
            proximal_node = tuple(edge["node2"])
        elif isinstance(edge["node1"], str):
            # Legacy
            current_node = tuple([float(coord) for coord in edge["node1"][1:-1].split(" ") if len(coord)>0])
            proximal_node = tuple([float(coord) for coord in edge["node2"][1:-1].split(" ") if len(coord)>0])

        if proximal_node in blackdict or random()<p:
            blackdict[current_node] = True
            continue

        radius_list.append(radius)
        thickness = radius * scale_factor
        edges.append([(current_node[axes[1]]/sim_shape[1],current_node[axes[0]]/sim_shape[0]),(proximal_node[axes[1]]/sim_shape[1],proximal_node[axes[0]]/sim_shape[0])])
        radii.append(thickness)
    if colorize is not None:
        colors=np.copy(np.array(radii))
        colors = colors/no_pixels_x/1.3*3
        if colorize=="continous":
            colors=np.minimum(colors/0.03,1)
        elif colorize=="dicrete":
            c_new = np.zeros_like(colors)
            c_new[colors<=0.01]=0.1
            c_new[(colors>0.01) & (colors<=0.02)]=0.5
            c_new[colors>0.02]=1
            colors=c_new
        else:
            raise NotImplementedError("Colorize only supports the options 'continous' or 'discrete'!")
        colors=cm.plasma(colors)
    else:
        colors="w"
    ax.add_collection(collections.LineCollection(edges, linewidths=radii, colors=colors, antialiaseds=True, capstyle="round"))
    figure.canvas.draw()
    data = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    plt.close(figure)

    if colorize:
        img_gray = np.array(img.astype(np.float32))
    else:
        img_gray = np.array(Image.fromarray(img).convert("L")).astype(np.uint16)
    return img_gray, blackdict


def getCrossSlice(p1: tuple[int], p2: tuple[int], radius: int, voxel_size: float=1, image_dim=(255, 251, 120), mode: Literal['tube', 'cuboid']='cuboid'):
    """
    Computes relevant indices in an image tensor that contain the line from p1 to p2 with the given radius.
    
    Paramters:
        - p1: 3D point in simulation space
        - p2: 3D point in simulation space
        - radius: radius of line in simulation space scale
        - voxel_size: The voxel size of the rendered image w.r.t. the simulation space
        - image_dim: shape of image tensor in voxels
        - mode: Type of indexing strategy that being used. 'tube' is more precise and better for long lines. 'cuboid' is faster to compute and better for short lines
    """
    if mode=='tube':
        p1_scaled = p1/voxel_size
        p1_voxel = np.floor(p1_scaled)
        dims = range(len(p1_voxel))
        line = (p2-p1)
        num_steps = np.linalg.norm(line) / voxel_size
        step_update = line / np.linalg.norm(line)
        voxel_offset = math.ceil(radius / voxel_size)
        indices = list()
        current_indices = np.array(list(itertools.product(*[
            [p1_scaled[d] + k for k in range(-voxel_offset, voxel_offset+2)] for d in dims
        ])))
        indices = {tuple(vox) for vox in floor(current_indices)}

        for i in range(math.ceil(num_steps)):
            current_indices = current_indices + step_update
            indices.update([tuple(vox) for vox in floor(current_indices)])

        indices = np.array(list(indices))
        indices = indices[np.all(indices>0, axis=1)]
        indices = indices[np.all(image_dim-indices>0, axis=1)]
        return indices

    if mode=='cuboid':
        voxel_offset = (radius / voxel_size) * math.sqrt(2)
        s_x,s_y,s_z = p1/voxel_size
        e_x,e_y,e_z = p2/voxel_size
        if s_x>e_x:
            e_x, s_x = s_x, e_x
        if s_y>e_y:
            e_y, s_y = s_y, e_y
        if s_z>e_z:
            e_z, s_z = s_z, e_z
        s_x = max(0, math.floor(s_x-voxel_offset))
        e_x = min(image_dim[0],math.ceil(e_x+voxel_offset+1))
        s_y = max(0, math.floor(s_y-voxel_offset))
        e_y = min(image_dim[1],math.ceil(e_y+voxel_offset+1))
        s_z = max(0, math.floor(s_z-voxel_offset))
        e_z = min(image_dim[2],math.ceil(e_z+voxel_offset+1))
        indices = np.array(list(itertools.product(
            list(range(s_x, e_x)),
            list(range(s_y, e_y)),
            list(range(s_z, e_z))
        )))
        return indices

    raise NotImplementedError(mode)


def voxelize_forest(forest: dict,
                    volume_dimensions: Sequence[float],
                    radius_list:list=None,
                    sim_space=None,
                    config=None,
                    min_radius=0,
                    max_radius=1,
                    max_dropout_prob=0,
                    blackdict: dict[str, bool]=None) -> Tuple[np.ndarray, dict[str, bool]]:
    """
    Converts the given 3D forest into a 3D (grayscale) volume.
    Antialiased drawing of the tree edges is performed manually.

    Parameters:
    -----------
        - forest: list of edges. An edge is a dictionary with 'node1' position 'node2' position and 'radius'.
        - volume_dimensions: Dimensions of the final 3D volume
        - MIP_axis: Axis along which to take the maximum intensity projection. Default is the z dimension
        - radius_list: A list to collect all edge radii. Default is None
        - min_radius: All edges with radius smaller than this will not be included in the grayscale image
        - max_radius: All edges with radius larger than this will not be included in the grayscale image
        - max_dropout_prob: Maximum probablity with which an edge and its decendents are dropped. 
                            The probabily is sampled from :math:`p = P**10, P ~ Uniform(0,max_dropout_prob)`
        - blackdict: A dictionary containing all parent nodes that were removed in the paired image.
                    All edges from these nodes will be dropped. If a dictionary is provided, no other edges will be removed.

    Returns:
    -------
        - 3D grayscale volume of all rendered vessels.
        - backlist dictionary containing all parent nodes that were dropped 
    """
    MIN_DIM_SIZE = 30  # A minimal size of 30 is necessary to consider nodes that accidentally grew outside of the simulation space.
    image_dim = np.array([max(MIN_DIM_SIZE,d) for d in volume_dimensions])
    if radius_list is None:
        radius_list=[]
    scale_factor = max(image_dim)
    pos_correction = (image_dim-np.array(volume_dimensions))//2
    no_voxel_x, no_voxel_y, no_voxel_z = image_dim
    voxel_size = 1
    voxel_diag = np.linalg.norm(np.array([1, 1, 1]))

    img = np.zeros((no_voxel_x, no_voxel_y, no_voxel_z))
    if config['mode'] == 'airways':
        lumen_img = np.zeros((no_voxel_x, no_voxel_y, no_voxel_z))

    if blackdict is None:
        blackdict = dict()
        p = random()**10 * max_dropout_prob
    else:
        p=0
    for edge in forest:
        radius = float(edge["radius"])
        if radius<min_radius or radius>max_radius:
            continue
        
        if isinstance(edge["node1"], np.ndarray) or isinstance(edge["node1"], list):
            current_node = tuple(edge["node1"])
            proximal_node = tuple(edge["node2"])
        elif isinstance(edge["node1"], str):
            # Legacy
            current_node = tuple([float(coord) for coord in edge["node1"][1:-1].split(" ") if len(coord)>0])
            proximal_node = tuple([float(coord) for coord in edge["node2"][1:-1].split(" ") if len(coord)>0])
        
        if proximal_node in blackdict or random()<p:
            blackdict[current_node] = True
            continue
        radius_list.append(radius)

        radius *= scale_factor
        current_node = np.array(current_node)*scale_factor
        proximal_node = np.array(proximal_node)*scale_factor

        voxel_indices = np.array(getCrossSlice(
            current_node+pos_correction, proximal_node+pos_correction, radius, voxel_size, image_dim
        ))
        if len(voxel_indices) == 0:
            continue
        indices = (voxel_indices+.5) * voxel_size

        # Calculate orthogonal projection of each voxel onto segment
        segment_vector = (current_node+pos_correction) - (proximal_node+pos_correction)
        voxel_vector = indices - (proximal_node+pos_correction)
        scalar_projection = np.dot(voxel_vector, segment_vector) / np.dot(segment_vector, segment_vector)
        inside_segment = np.logical_and(scalar_projection > 0, scalar_projection < 1)

        # If the projection falls onto the segment, add the vessel's contribution to the oxygen map
        vector_projection = (proximal_node+pos_correction) + np.dot(scalar_projection[:, None], segment_vector[None, :])
        dist = np.linalg.norm(indices - vector_projection, axis=1)

        inds: list[list] = voxel_indices[inside_segment].astype(np.uint16).transpose().tolist()
        volume_contribution = 1 - ((dist[inside_segment] - (radius - voxel_diag/2)) / voxel_diag)
        
        img[tuple(inds)] = np.maximum(volume_contribution, img[tuple(inds)])
        # Handle beginning and end
        dist = np.minimum(
            np.linalg.norm(indices-(current_node+pos_correction), axis=1),
            np.linalg.norm(indices-(proximal_node+pos_correction), axis=1)
        )
        inds = voxel_indices.astype(np.uint16).transpose().tolist()
        img[tuple(inds)] = np.maximum(1-((dist - (radius - voxel_diag/2)) / voxel_diag), img[tuple(inds)])
        
        if config['mode'] == 'airways':
            lumen_radius = 0.6 * radius
            lumen_voxel_indices = np.array(getCrossSlice(
                current_node+pos_correction, proximal_node+pos_correction, lumen_radius, voxel_size, image_dim
            ))
            lumen_indices = (lumen_voxel_indices+.5) * voxel_size
            
            lumen_voxel_vector = lumen_indices - (proximal_node+pos_correction)
            lumen_scalar_projection = np.dot(lumen_voxel_vector, segment_vector) / np.dot(segment_vector, segment_vector)
            lumen_inside_segment = np.logical_and(lumen_scalar_projection > 0, lumen_scalar_projection < 1)
                    
            lumen_vector_projection = (proximal_node+pos_correction) + np.dot(lumen_scalar_projection[:, None], segment_vector[None, :])
            lumen_dist = np.linalg.norm(lumen_indices - lumen_vector_projection, axis=1)
            
            lumen_inds: list[list] = lumen_voxel_indices[lumen_inside_segment].astype(np.uint16).transpose().tolist()
            lumen_volume_contribution = 1 - ((lumen_dist[lumen_inside_segment] - (lumen_radius - voxel_diag/2)) / voxel_diag)
        
            lumen_img[tuple(lumen_inds)] = np.maximum(lumen_volume_contribution, lumen_img[tuple(lumen_inds)])
            lumen_dist = np.minimum(
                np.linalg.norm(lumen_indices-(current_node+pos_correction), axis=1),
                np.linalg.norm(lumen_indices-(proximal_node+pos_correction), axis=1)
            )
            lumen_inds = lumen_voxel_indices.astype(np.uint16).transpose().tolist()
            lumen_img[tuple(lumen_inds)] = np.maximum(1-((lumen_dist - (lumen_radius - voxel_diag/2)) / voxel_diag), lumen_img[tuple(lumen_inds)])
    
    out_dict = {}       
    img=img[pos_correction[0]:pos_correction[0]+volume_dimensions[0],
            pos_correction[1]:pos_correction[1]+volume_dimensions[1],
            pos_correction[2]:pos_correction[2]+volume_dimensions[2]]        
    
    seg = np.zeros_like(img)
    if config['mode'] == 'airways':
        lumen_img=lumen_img[pos_correction[0]:pos_correction[0]+volume_dimensions[0],
                            pos_correction[1]:pos_correction[1]+volume_dimensions[1],
                            pos_correction[2]:pos_correction[2]+volume_dimensions[2]]
        
        seg[lumen_img > 0.2] = 1
        wall = np.zeros_like(img)
        wall = img - lumen_img*1.6
        lumen_img = np.clip(lumen_img, 0, 1)
        wall = np.clip(wall, 0, 1)
        out_dict['lumen_img'] = lumen_img.astype(np.float32)
        out_dict['wall'] = wall.astype(np.float32)
    else:
        seg[img > 0.2] = 1     
    img = np.clip(img, 0, 1) 
    if sim_space is not None:
        lobe = sim_space.geometry
    
    out_dict['img'] = img.astype(np.float32)
    out_dict['seg'] = seg.astype(np.uint8)
    out_dict['blackdict'] = blackdict
    out_dict['lobe'] = lobe.astype(np.uint8)
    
    return out_dict


def save_2d_img(img: np.ndarray, out_dir: str, name: str):
    """
    Saves a prevously rendered image at the given folder. 

    Parameters:
    -----------
        - img: numpy array of the rendered image. Use the rasterize_forest function to create the image.
        - out_dir: Path to the output folder
        - name: Name of the image
    """
    Image.fromarray(img.astype(np.uint8)).save(f'{out_dir}/{name}.png')


def plot_vessel_radii(out_dir: str, radius_list: list[float] = []):
    """
    Creates a histogram of all simulated vessel radii and saves it as an image.

    Parameters:
    -----------
        - out_dir: Path to the output folder
        - radius_list: List of all simulated radii
    """
    plt.figure()
    bins = np.linspace(min(radius_list), max(radius_list),40)
    plt.xlim([min(radius_list), max(radius_list)])

    plt.hist(radius_list, bins=bins, alpha=0.5)
    plt.title('Vessel Radii Distribution')
    plt.xlabel('Radius')
    plt.ylabel('Count')
    plt.gca().set_yscale('log')

    plt.savefig(f"{out_dir}/hist.png", bbox_inches="tight")
    plt.close()
