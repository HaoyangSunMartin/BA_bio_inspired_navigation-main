from plotting.plotThesis import *
import os
import numpy as np


class BlockCell:
    """Class to keep track of an individual Block Cell"""
    def __init__(self, gc_connections, coordinates):
        self.gc_connections = gc_connections  # Connection matrix to grid cells of all modules; has form (n^2 x M)
        self.env_coordinates = coordinates  # Save x and y coordinate at moment of creation

        self.plotted_found = [False, False]  # Was used for debug plotting, of linear lookahead

    def compute_firing(self, s_vectors):
        """Computes firing value based on current grid cell spiking"""
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # determine where connection exist to grid cells
        filtered = np.multiply(gc_connections, s_vectors)  # filter current grid cell spiking, by connections
        modules_firing = np.sum(filtered, axis=1) / np.sum(s_vectors, axis=1)  # for each module determine bc firing
        firing = np.average(modules_firing)  # compute overall bc firing by summing averaging over modules
        return firing

    def compute_firing_2x(self, s_vectors, axis, plot=False):
        """Computes firing projected on one axis, based on current grid cell spiking"""
        new_dim = int(np.sqrt(len(s_vectors[0])))  # n

        s_vectors = np.where(s_vectors > 0.1, 1, 0)  # mute weak grid cell spiking, transform to binary vector
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # mute weak connections, transform to binary vector

        proj_s_vectors = np.empty((len(s_vectors[:, 0]), new_dim))
        for i, s in enumerate(s_vectors):
            s = np.reshape(s, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
            proj_s_vectors[i] = np.sum(s, axis=axis)  # sum over column/row

        proj_gc_connections = np.empty_like(proj_s_vectors)
        for i, gc_vector in enumerate(gc_connections):
            gc_vector = np.reshape(gc_vector, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
            proj_gc_connections[i] = np.sum(gc_vector, axis=axis)  # sum over column/row

        filtered = np.multiply(proj_gc_connections, proj_s_vectors)  # filter projected firing, by projected connections

        norm = np.sum(np.multiply(proj_s_vectors, proj_s_vectors), axis=1)  # compute unnormed firing at optimal case

        firing = 0
        modules_firing = 0
        for idx, filtered_vector in enumerate(filtered):
            # We have to distinguish between modules tuned for x direction and modules tuned for y direction
            if np.amin(filtered_vector) == 0:
                # If tuned for right direction there will be clearly distinguishable spikes
                firing = firing + np.sum(filtered_vector) / norm[idx]  # normalize firing and add to firing
                modules_firing = modules_firing + 1

        firing = firing/modules_firing  # divide by modules that we considered to get overall firing

        # Plotting options, used for linear lookahead debugging
        if plot:
            for idx, s_vector in enumerate(s_vectors):
                plot_vectors(s_vectors[idx], gc_connections[idx], axis=axis, i=idx)
            plot_linear_lookahead_function(proj_gc_connections, proj_s_vectors, filtered, axis=axis)

        if firing > 0.97 and not self.plotted_found[axis]:
            for idx, s_vector in enumerate(s_vectors):
                plot_vectors(s_vectors[idx], gc_connections[idx], axis=axis, i=idx, found=True)
            plot_linear_lookahead_function(proj_gc_connections, proj_s_vectors, filtered, axis=axis, found=True)
            self.plotted_found[axis] = True

        return firing


class BlockCellList:
    """A PlaceCellNetwork holds information about all Place Cells"""

    def __init__(self, from_data):
        self.block_cells = []  # array of block cells(all isolated)
        self.threshold = 0.86
        self.block_distance = 1.3
        if not from_data:
            # Load place cells if wanted
            gc_connections = np.load("data/pc_model/gc_connections.npy")
            env_coordinates = np.load("data/pc_model/env_coordinates.npy")

            for idx, gc_connection in enumerate(gc_connections):
                bc = BlockCell(gc_connection)
                bc.env_coordinates = env_coordinates[idx]
                self.block_cells.append(bc)


    def create_new_bc(self, gc_modules, coordinate):
        # Consolidate grid cell spiking vectors to matrix of size n^2 x M
        s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
        for m, gc in enumerate(gc_modules):
            s_vectors[m] = gc.s
        weights = np.array(s_vectors)
        bc = BlockCell(weights, coordinate)
        self.block_cells.append(bc)

    ###changes by Haoyang Sun - start
    def check_current_BC(self, gc_modules):

        firing_values = self.compute_firing_values(gc_modules)
        if len(firing_values) == 0:
            return -1
        max_firing = max(firing_values)
        idx = firing_values.index(max_firing)
        if max_firing < self.threshold:
            return -1
        else:
            return idx

    ###changes by Haoyang Sun - end

    def track_movement(self, gc_network,env, generate_BC=False):
        """Keeps track of :
        current grid cell module firing
        obstacles in the environment -> decide whether to create new Block Cells
        this track_movement function returns the idx and coordinates of the created block cell
        """
        gc_modules = gc_network.gc_modules
        ##get the obstacle datas from the environment module
        dire_vectors = env.get_blocked_directions(self.block_distance)
        if len(dire_vectors) > 0:
            print("blocked vectors are", dire_vectors)
        gc_network.reset_s_virtual()
        created_new_bc = False
        for n, vector in enumerate(dire_vectors):
            vector_slow = np.multiply(vector,[0.5,0.5]) #decrease the speed to keep the GC network stable
            gc_network.track_movement(vector_slow, virtual=True, dt_alternative=2)
            firing_values = self.compute_firing_values(gc_network.gc_modules, virtual=True)
            # create a new block cell if there is no block cell present there

            if len(firing_values) == 0 or max(firing_values) < self.threshold:
                coordinates = np.array(env.xy_coordinates[-1]+vector)
                print("creating new block cell")
                self.create_new_bc(gc_modules, coordinates)
                print("current block cell list is: ", len(self.block_cells), "coordinate is: ", coordinates)
                created_new_bc = True
                if len(self.block_cells) % 10 ==0 and len(self.block_cells)!= 0 :
                    plot_env_map_with_bc(self, env)
            else:
                print(vector," is close to BC Nr. ", np.argmax(firing_values), "discard")

            gc_network.reset_s_virtual()
        if created_new_bc:
            return [created_new_bc, coordinates]
        else:
            return [False, None]

    def compute_firing_values(self, gc_modules, virtual=False, axis=None, plot=False):
        s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
        # Consolidate grid cell spiking vectors that we want to consider
        for m, gc in enumerate(gc_modules):
            if virtual:
                s_vectors[m] = gc.s_virtual
            else:
                s_vectors[m] = gc.s

        firing_values = []
        for i, bc in enumerate(self.block_cells):
            if axis is not None:
                plot = plot if i == 0 else False  # linear lookahead debugging plotting
                firing = bc.compute_firing_2x(s_vectors, axis, plot=plot)  # firing along axis
            else:
                firing = bc.compute_firing(s_vectors)  # overall firing

            firing_values.append(firing)
        return firing_values

    def save_bc_list(self, filename=""):
        gc_connections = []
        env_coordinates = []
        for bc in self.block_cells:
            gc_connections.append(bc.gc_connections)
            env_coordinates.append(bc.env_coordinates)

        directory = "data/bc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/pc_model/gc_connections" + filename + ".npy", gc_connections)
        np.save("data/pc_model/env_coordinates" + filename + ".npy", env_coordinates)

###changes by Haoyang Sun-end