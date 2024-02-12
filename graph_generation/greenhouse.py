import random
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from typing import Tuple

from graph_generation.forest import Forest
from graph_generation.simulation_space import SimulationSpace
from graph_generation.utilities import eukledian_dist, norm_vector, normalize_vector, get_angle_between_vectors, get_angle_between_two_vectors
from graph_generation.tree import Node
from graph_generation.element_mesh import CoordKdTree, NodeKdTree, SpacePartitioner
from tqdm import tqdm
import time

class Greenhouse():

    def __init__(self, config: dict):
        self.config = config
        self.modes: list[dict] = config['modes']
        
        self.sigma_t: float = 1
        self.param_scale: float = config['param_scale']
        self.d: float = config['d'] / self.param_scale
        self.r: float = config['r'] / self.param_scale
        self.simspace = SimulationSpace(config["SimulationSpace"])

        self.init_params_from_config(self.modes[0])

    def init_params_from_config(self, config: dict):
        self.I: int = config['I']
        self.N: int = config['N']
        self.eps_n: float = config['eps_n']
        self.eps_s: float = config['eps_s']
        self.eps_k: float = config['eps_k']
        self.delta: float = config['delta']
        self.gamma: float = config['gamma']
        self.phi: float = config['phi']
        self.omega: float = config['omega']
        self.kappa: float = config['kappa']
        self.delta_sigma: float = config['delta_sigma']
        self.sigma_t: float = 1
        
        self.orig_scale = [param / self.param_scale for param in [self.eps_k, self.eps_n, self.eps_s, self.delta]]
        self.orig_scale.append(self.d)

    def set_forests(self, forest: Forest):
        self.forest = forest

    def develop_forest(self):
        """
        Main loop. Generates the airway/vessel forest
        """
        self.node_mesh = NodeKdTree()
        self.node_mesh.extend(list(self.forest.get_nodes()))
        self.active_node_mesh = NodeKdTree()
        self.active_node_mesh.extend(list(self.forest.get_nodes()))

        # Stats
        self.nodes_per_step = [0]
        self.oxys_per_step = [0]
        self.time_per_step = []

        self.oxy_mesh = CoordKdTree()
        t = 0
        for mode in self.modes:
            if mode["name"] != self.modes[0]["name"]:
                self.init_params_from_config(mode)
            if self.I<=0:
                continue

            for t in range(t,t+self.I):
                s = time.time()
                # 1. Sample attraction points
                self.sample_oxygen_sinks(int(self.N), max(self.eps_n, self.eps_k), self.eps_s, t=t)
                # 2. airway/vessel growth
                new_nodes = self.grow_vessels(self.active_node_mesh, self.oxy_mesh, self.gamma, self.delta, first_mode=mode == self.modes[0], t=t)
                self.node_mesh.extend(new_nodes)
                self.active_node_mesh.extend(new_nodes)
                # 3. All attraction points within distance d_suff of airway/vessel nodes are converted to carbon-dioxid sources
                to_remove = set()
                to_add = set()
                for node in new_nodes:
                    for oxy in self.oxy_mesh.find_elements_in_distance(node.position, self.eps_k):
                        # Oxygen sink is satisfied. Transform to CO2 source
                        to_remove.add(oxy)
                        # self.oxy_mesh.delete(oxy)
                self.oxy_mesh.delete_all(to_remove)

                # 4. Scaling
                self.simulation_space_expansion()

                # Update stats
                self.time_per_step.append(time.time()-s)
                self.nodes_per_step.append(len(self.node_mesh.get_all_elements()))
                self.oxys_per_step.append(len(self.oxy_mesh.get_all_elements()))


    def simulation_space_expansion(self):
        """
        Scales all distance related parameters to simulate the expansion of the simulation space.
        This is motivated by the growth of tissue in real life.
        """
        # scaling factor at time t: σ_t = σ_t−1 +∆σ
        self.sigma_t = self.sigma_t + self.delta_sigma
        self.eps_k, self.eps_n, self.eps_s, self.delta, self.d = [param / self.sigma_t for param in self.orig_scale]
        self.d = max(self.d, 0.04/self.param_scale)
        
        self.node_mesh.reassign(self.delta)
        self.active_node_mesh.reassign(self.delta)
        self.oxy_mesh.reassign(self.delta)
    
    def grow_vessels(self, node_mesh: SpacePartitioner[Node], att_mesh: SpacePartitioner, gamma: float, delta: float, first_mode=True, t=0) -> list[Node]:
        """
        Performs airway/vessel growth

        Paramters:
        ---------
        - node_mesh: NodeMesh of vessel nodes that are grown
        - att_mesh: CoordsMesh of oxygen or co2 sinks

        Returns:
        --------
        List of all new Nodes that were added in this iteration
        """
        # Nearest Neighbor search
        att_node_assignment: dict[Node, list[tuple[float]]] = self.assign_attraction_points_to_node(node_mesh, att_mesh, delta)
        new_nodes: list[Node] = []
        # Vessel Growth
        for node, atts in att_node_assignment.items():
            # vector_to_center = np.array(self.FAZ_center)-node.position[:2]
            # dist_to_center = np.linalg.norm(vector_to_center)
            if node.is_leaf:
                v = node.get_proximal_segment()
                angles_i = get_angle_between_vectors(v, atts - node.position)
                # Requirements for oxygen sink with position p_0, distance r and angle θ
                #   - r = ||p_o − p_j||<=δ
                #   - θ = cos^−1 e_ij * nrm(p_o − p_j) <= γ/2
                valid_inds = angles_i <= max(gamma/2,0)
                atts = np.array(atts)[valid_inds]
                if len(atts) == 0:
                    continue
                avg_attraction_vector = sum([norm_vector(att-node.position) for att in atts])

                angles = angles_i[valid_inds]
                # IFF the standard deviation of the angles formed by the attraction vectors is larger than a predefined threshold φ
                if np.std(angles) > self.phi:
                    # Bifurcation:

                    # Radii with fix terminal length r
                    r_1 = r_2 = self.r
                    r_p = (r_1**self.kappa + r_2**self.kappa)**(1/self.kappa)

                    d1 = d2 = self.d

                    # Its bifurcation angle is dictated by Murphy's law
                    phi_1 = np.degrees(np.arccos((r_p ** 4 + r_1 ** 4 - r_2 ** 4) / (2 * r_p ** 2 * r_1 ** 2)))
                    phi_2 = np.degrees(np.arccos((r_p ** 4 + r_2 ** 4 - r_1 ** 4) / (2 * r_p ** 2 * r_2 ** 2)))

                    c = np.mean(atts, axis=0)

                    d_parent_c = normalize_vector(c-node.position)

                    X = np.array([oxy-c for oxy in atts]).transpose()
                    X_cov = np.cov(X)
                    w, v = np.linalg.eig(X_cov)
                    d_l = v[:, np.argmax(w)]

                    p_new_1 = np.real(node.position + norm_vector(np.cos(np.radians(phi_1)) * d_parent_c + np.sin(np.radians(phi_1))*d_l) * d1)
                    p_new_2 = np.real(node.position + norm_vector(np.cos(np.radians(phi_2)) * d_parent_c - np.sin(np.radians(phi_2))*d_l) * d2)

                    new_nodes.append(node.tree.add_node(p_new_1, r_1, node, self.kappa))
                    new_nodes.append(node.tree.add_node(p_new_2, r_2, node, self.kappa))
                    # Update radii of all parent edges up to root with Murray
                    node.optimize_edge_radius_to_root()
                    node_mesh.delete(node)
                else:
                    # Elongation
                    g = self.omega*norm_vector(v) + (1-self.omega)*norm_vector(sum([norm_vector(att-node.position) for att in atts]))    

                    p_k = np.real(node.position + self.d * norm_vector(g))
                    new_nodes.append(node.tree.add_node(p_k, self.r, node, self.kappa))
            elif node.is_inter_node:
                # Calculate optimal radius and angle with Murray
                r_p = node.get_proximal_radius()
                r_1 = node.get_distal_radius()

                r_2 = self.r
                r_p = (r_1**self.kappa + r_2**self.kappa)**(1/self.kappa)
                phi_1 = np.degrees(np.arccos((r_p ** 4 + r_1 ** 4 - r_2 ** 4) / (2 * r_p ** 2 * r_1 ** 2)))
                phi_2 = np.degrees(np.arccos((r_p ** 4 + r_2 ** 4 - r_1 ** 4) / (2 * r_p ** 2 * r_2 ** 2)))
                
                # Reqirements for oxygen sinks:
                #   - α−γ/2 <= θ <= α+γ/2
                #   - r ≤ δ
                angles_distal = get_angle_between_vectors(node.get_distal_segment(0), atts - node.position)
                angles_proximal = get_angle_between_vectors(node.get_proximal_segment(), atts - node.position)
                atts = np.array(atts)[
                    (phi_1 + phi_2 - gamma/2 <= angles_distal) &
                    (angles_distal <= (phi_1 + phi_2 + gamma/2)) &
                    (angles_proximal<= phi_2 + gamma/2)
                ]
                if len(atts) == 0:
                    continue

                avg_attraction_vector = sum([norm_vector(att-node.position) for att in atts])
                # Rodrigues' rotation formula
                # If v is a vector in ℝ3 and k is a unit vector describing an axis of rotation about which 
                # v rotates by an angle θ according to the right hand rule, the Rodrigues formula for the rotated vector vrot is 
                # v_rot = v*cos(θ) + (k×v)sin(θ) + k(k·v)(1-cos(θ))
                # Rotation axis is cross product of child-vector and average attraction vector
                distal_vector = norm_vector(node.get_distal_segment())
                cross = np.cross(distal_vector, avg_attraction_vector)
                if all(cross==0) : #or ((dist_to_center / (2*self.FAZ_radius))**5 <= random.uniform(0,1) and get_angle_between_two_vectors(vector_to_center, avg_attraction_vector[:2])<=90):
                    continue
                rot_axis = norm_vector(cross)
                theta = phi_2 #get_angle_between_vectors(distal_vector, avg_attraction_vector)
                # Calculate hypothetical optimal branch closest to the average attraction vector
                v = distal_vector*np.cos(np.radians(theta)) + np.cross(rot_axis, distal_vector)*np.sin(np.radians(theta)) \
                        + rot_axis*np.dot(rot_axis, distal_vector)*(1-np.cos(np.radians(theta)))
                # if get_angle_between_vectors(v, node.get_proximal_segment()) > 180:
                #     v = -v
                g = self.omega*norm_vector(v) + (1-self.omega)*norm_vector(avg_attraction_vector)
                
                d = self.d

                p_k = np.real(node.position + d * norm_vector(g))
                new_nodes.append(node.tree.add_node(p_k, self.r, node, self.kappa))
                # Update raddi of all parent edges up to root with Murray
                node.optimize_edge_radius_to_root()
                node_mesh.delete(node)
        return new_nodes

    def _calculate_oxygen_distance(self, r): 
        """
        Models the oxygen concentration heuristic by Schneider et al., 2012 (https://doi.org/10.1016/j.media.2012.04.009)
        """
        c_oxygen = 203.9e-3 # oxygen concentration inside vessel lumen in m^3 per m^3
        kappa = 0.02 * c_oxygen # peak perfusion level in simulation space
        r0 = 3.5e-3 # vessel radius perfusing maximum amount of oxygen in mm
        c1 = kappa * (r*self.param_scale/r0)*np.exp(1-(r*self.param_scale/r0))
        return c1 * 6 / self.param_scale

    def sample_oxygen_sinks(self, N=1000, eps_n=0.04, eps_s=0.3, t=0) -> list[tuple[float]]:
        """
        Sample Oyigen sinks from hypoxic tissue.

        Parameters:
        -----------
        - N: Number of tries
        - eps_n: threshold to airway/vessel node within which samples are discarded
        - eps_s: threshold to other oxigen sinks within which samples are discarded
        - radius: Radius in the center of the simulation space where no oxygen sinks are sampled from

        Returns:
        -------
        List of 3D coordinates of valid oxygen sinks
        """
        to_add = list()
        candidate_sinks = self.simspace.get_candidate_sinks(N)
        for candidate_sink in candidate_sinks:
            if all([eukledian_dist(candidate_sink, node.position) > self._calculate_oxygen_distance(node.radius) for node in self.node_mesh.find_elements_in_distance(candidate_sink, eps_n)]) \
                and self.oxy_mesh.find_nearest_element(candidate_sink, eps_s) is None \
                and (not to_add or all(np.linalg.norm(np.array(candidate_sink)-np.array(to_add), axis=1) > eps_s)):
                to_add.append(tuple(candidate_sink))
        self.oxy_mesh.extend(to_add)

    def assign_attraction_points_to_node(self, node_mesh: SpacePartitioner[Node], attraction_point_mesh: SpacePartitioner, delta: float) -> dict[Node, list[tuple[float]]]:
        """
        Performs nearest neighbor search to assign each attraction point to its closest vessel node
        
        Paramters:
        ---------
        node_mesh: NodeMesh of nodes of interest
        attraction_point_mesh: CoordsMesh of oxygen or co2 sinks of interest

        Returns:
        -------
        Node to attraction points dictionary where each attraction point is assigned to its closest node.
        List where attractions points at index i are closest to node i in forest
        """
        assignment = dict()
        for attraction_point in attraction_point_mesh.get_all_elements():
            closest = node_mesh.find_nearest_element(attraction_point, max_dist=delta)
            if closest is None:
                continue
            if closest in assignment:
                assignment[closest].append(attraction_point)
            else:
                assignment[closest] = [attraction_point]
        return assignment

    def save_forest_fig(self, output_path='output.png'):
        fig = plt.figure(figsize=((12, 12)))
        ax = plt.axes(projection='3d')
        forests = [self.forest]
        for forest in forests:
            for tree in  tqdm(forest.get_trees(), desc="Preparing figure"):
                size_x = tree.size_x
                size_y = tree.size_y
                size_z = tree.size_z

                edges = np.array([np.concatenate([node.parent.position, node.position]) for node in tree.get_tree_iterator(exclude_root=True)])
                radii = np.array([node.radius for node in tree.get_tree_iterator(exclude_root=True)])
                radii /= radii.max()

                linewidth = 8

                for edge, radius in zip(edges, radii):
                    plt.plot([edge[0], edge[3]], [edge[1], edge[4]], [edge[2], edge[5]], c=plt.cm.jet(radius), linewidth=linewidth * radius, axes=ax)

                ax.set_xlim(0, size_x)
                ax.set_ylim(0, size_y)
                ax.set_zlim(0, size_z)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

                ax.view_init(elev=90.0, azim=0.0)
                ax.set_box_aspect((size_x, size_y, size_z))
        plt.savefig(output_path, bbox_inches="tight")
            
    def save_stats(self, out_dir: str):
        plt.figure(figsize=(6,6))
        oxys = np.array(self.oxy_mesh.get_all_elements())
        plt.plot(oxys[:,0], oxys[:,1], 'r.')
        plt.title('Final Oxygen Sink Distribution')
        plt.savefig(f'{out_dir}/oxy_distribution.png', bbox_inches='tight')
        plt.cla()

        plt.plot(self.time_per_step)
        total = time.strftime('%H:%M:%S', time.gmtime(sum(self.time_per_step)))
        plt.title(f'Runtime Per Iteration (Total={total})')
        plt.xlabel("Iterations")
        plt.ylabel("Seconds")
        plt.savefig(f'{out_dir}/time_per_step.png', bbox_inches='tight')
        plt.cla()

        plt.plot(self.nodes_per_step)
        plt.plot(self.oxys_per_step)
        plt.legend(['Nodes', 'Oxygen Sinks'])
        plt.title('Growth Over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Amount')
        plt.savefig(f'{out_dir}/growth_over_time.png', bbox_inches='tight')
        plt.close()
