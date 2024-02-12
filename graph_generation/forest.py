import os
import random
import math
from typing import Generator

import numpy as np

from graph_generation.tree import Tree, Node
from graph_generation.simulation_space import SimulationSpace
from graph_generation.utilities import norm_vector
import csv


class Forest:

    def __init__(self, config: dict, d_0: float, r_0: float, sim_space: SimulationSpace):
        """
        Initialize a forest of multiple vessel trees.

        Parameters:
        ----------
            - config: forest configuration dictionary
            - d_0: Initial vessel length used for the root stumps.
            - r_0: Initial radius used for the root stumps.
            - sim_space: Simulation space in which the forest in grown.
        """
        self.trees: list[Tree] = []
        self.sim_space = sim_space
        self.size_x, self.size_y, self.size_z = self.sim_space.shape
        if config['type'] == 'bronchi':
            self._initialize_tree_stumps_from_bronchi(config, d_0, r_0)
        else:
            raise NotImplementedError(f"The Forest initialization type '{config['type']}' is not implemented.")

    def _initialize_tree_stumps_from_bronchi(self, config, d_0: float, r_0: float):
        roots = config['roots']
        for root in roots:
            tree_name = f'AirwayTree_{root["name"]}'
            position = np.array(root['position'])
            position /= self.sim_space.geometry_size
            direction = np.array(root['direction'])
            direction = direction / np.linalg.norm(direction) * d_0

            tree = Tree(tree_name, tuple(position), r_0, self.size_x, self.size_y, self.size_z, self)
            tree.add_node(position=tuple(position + direction), radius=r_0, parent=tree.root)
            self.trees.append(tree)

    def get_trees(self):
        return self.trees

    def get_nodes(self) -> Generator[Node, None, None]:
        for tree in self.trees:
            for node in tree.get_tree_iterator(exclude_root=False, only_active=False):
                yield node

    def get_node_coords(self) -> Generator[np.ndarray, None, None]:
        for tree in self.trees:
            for node in tree.get_tree_iterator(exclude_root=False, only_active=False):
                yield node.position

    def save(self, save_directory='.'):
        name = 'Forest'
        os.makedirs(save_directory, exist_ok=True)
        filepath = os.path.join(save_directory, name + '.csv')
        with open(filepath, 'w+') as file:
            writer = csv.writer(file)
            writer.writerow(["node1", "node2", "radius"])
            for tree in self.get_trees():
                for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                    proximal_node = current_node.get_proximal_node()
                    radius = current_node.radius
                    writer.writerow([current_node.position, proximal_node.position, radius])
