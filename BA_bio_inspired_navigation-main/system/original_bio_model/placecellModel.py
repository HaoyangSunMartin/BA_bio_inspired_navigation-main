from plotting.plotThesis import *
import os
import numpy as np


class PlaceCell:
    """Class to keep track of an individual Place Cell"""
    def __init__(self, gc_connections):
        self.gc_connections = gc_connections  # Connection matrix to grid cells of all modules; has form (n^2 x M)
        self.env_coordinates = None  # Save x and y coordinate at moment of creation

        self.plotted_found = [False, False]  # Was used for debug plotting, of linear lookahead

    def show_connections(self):
        print(self.gc_connections.shape)
        plot_gc_vectors(self.gc_connections)

    def compute_firing(self, s_vectors):
        """Computes firing value based on current grid cell spiking"""
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # determine where connection exist to grid cells
        filtered = np.multiply(gc_connections, s_vectors)  # filter current grid cell spiking, by connections
        #note that the shape of the filtered is (6,1600)
        modules_firing = np.sum(filtered, axis=1) / np.sum(s_vectors, axis=1)  # for each module determine pc firing

        firing = np.average(modules_firing)  # compute overall pc firing by summing averaging over modules
        return firing

    def compute_firing_2x(self, s_vectors, axis, tuned_vector=None, gm_vector=None, plot=False):
        """Computes firing projected on one axis, based on current grid cell spiking"""
        new_dim = int(np.sqrt(len(s_vectors[0])))  # n
        # filter the neural sheet of the gc connection and the gc modules
        s_vectors = np.where(s_vectors > 0.1, 1, 0)  # mute weak grid cell spiking, transform to binary vector
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # mute weak connections, transform to binary vector
        # check the tuned direction of the gc connection and the gc modules
        proj_s_vectors = np.empty((len(s_vectors[:, 0]), new_dim))
        direction_s_vector = []  # record the tuned direction of the s_vector
        for i, s in enumerate(s_vectors):
            s = np.reshape(s, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
            p = np.sum(s, axis=axis)  # sum over column/row
            proj_s_vectors[i] = p
            if np.amin(p) == 0:  # if the gc connection is tuned to this direction, record 1 else 0
                direction_s_vector.append(1)
            else:
                direction_s_vector.append(0)
        proj_gc_connections = np.empty_like(proj_s_vectors)
        direction_connection_vector = []  # record the tuned direction of the gc_connection
        for i, gc_vector in enumerate(gc_connections):
            gc_vector = np.reshape(gc_vector, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
            p = np.sum(gc_vector, axis=axis)  # sum over column/row
            proj_gc_connections[i] = p
            if np.amin(p) == 0:  # if the gc connection is tuned to this direction, record 1 else 0
                direction_connection_vector.append(1)
            else:
                direction_connection_vector.append(0)

        # only calculate firing basing on the aligned gc modules to this direction
        filtered = np.multiply(proj_s_vectors, proj_gc_connections)
        firing = 0.0
        num_firing = 0
        normal = np.multiply(proj_gc_connections, proj_gc_connections)
        set = [0, 1, 2, 3, 4, 5, 6]
        # calculate weight of each module:
        direction_norm = 0.0
        same_weight = False
        weight = []

        for i, k in enumerate(direction_s_vector):
            if k == 1 and direction_connection_vector[i] == 1:
                direction_norm = direction_norm + (1.0 / gm_vector[i])

        for i, g in enumerate(gm_vector):
            if direction_connection_vector[i] == 1 and direction_s_vector[i] == 1:
                w = (1.0 / gm_vector[i]) / direction_norm
            else:
                w = 0.0
            weight.append(w)

        if same_weight:
            weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # print( "the direction s vector of ",self.env_coordinates," on axis ", axis, " is ",direction_s_vector)
        # print(direction_connection_vector)
        # print(weight)
        for i, f in enumerate(filtered):
            if direction_connection_vector[i] == 1 and direction_s_vector[i] == 1 and i in set:
                firing += weight[i] * (np.sum(f) / np.sum(normal[i]))

                num_firing += 1
        # print(firing)
        if same_weight:
            firing = firing / num_firing

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


def compute_weights(s_vectors):

    # weights = np.where(s_vectors > 0.1, 1, 0)
    weights = np.array(s_vectors)  # decided to not change anything here, but do it when computing firing

    return weights


class PlaceCellNetwork:
    """A PlaceCellNetwork holds information about all Place Cells"""
    def __init__(self, from_data=False, CUDA =False):
        self.place_cells = []  # array of place cells

        self.CUDA = CUDA
        if self.CUDA:
            self.path = "data/CUDA/pc_model/"
        else:
            self.path = "data/LINEAR/pc_model/"
        if from_data:


            # Load place cells if wanted
            gc_connections = np.load(self.path+"gc_connections.npy")
            env_coordinates = np.load(self.path+"env_coordinates.npy")

            for idx, gc_connection in enumerate(gc_connections):
                pc = PlaceCell(gc_connection)
                pc.env_coordinates = env_coordinates[idx]
                self.place_cells.append(pc)

    def create_new_pc(self, gc_modules):
        # Consolidate grid cell spiking vectors to matrix of size n^2 x M
        if gc_modules[0].CUDA:
            s_vectors = np.empty((len(gc_modules), gc_modules[0].sheetDimension[0]*gc_modules[0].sheetDimension[1]))
            for m, gc in enumerate(gc_modules):
                s_vectors[m] = gc.get_s().flatten()
            weights = compute_weights(s_vectors)
            pc = PlaceCell(weights)
            self.place_cells.append(pc)
        else:
            s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
            for m, gc in enumerate(gc_modules):
                s_vectors[m] = gc.s
            weights = compute_weights(s_vectors)
            pc = PlaceCell(weights)
            self.place_cells.append(pc)



    ###changes by Haoyang Sun - start
    def check_current_PC(self, gc_modules):

        firing_values = self.compute_firing_values(gc_modules)
        if len(firing_values) == 0:
            return -1
        max_firing = max(firing_values)
        idx = firing_values.index(max_firing)
        if max_firing < 0.85:
            return -1
        else:
            return idx
    ###changes by Haoyang Sun - end

    def track_movement(self, gc_modules, reward_first_found, generate_new_PC = False):
        """Keeps track of current grid cell firing"""

        firing_values = self.compute_firing_values(gc_modules)

        created_new_pc = False
        ###changes by Haoyang Sun - start
        current_PC = self.check_current_PC(gc_modules)
        if generate_new_PC == False:
            return [firing_values, False, current_PC]
        ###changes by Haoyang Sun - end

        #if len(firing_values) == 0 or np.max(firing_values) < 0.85 or reward_first_found:
        if current_PC == -1 or reward_first_found:
            # No place cell shows significant excitement
            # If the random number is above a threshold a new pc is created
            self.create_new_pc(gc_modules)
            firing_values.append(1)
            creation_str = "Created new place cell with idx: " + str(len(self.place_cells) - 1) \
                           + " | Highest alternative spiking value: " + str(np.max(firing_values))
            # print(creation_str)
            # if reward_first_found:
            #     print("Found the goal and created place cell")
            created_new_pc = True
        ###changes by Haoyang Sun - start
            current_PC = len(self.place_cells) - 1
        ###changes by Haoyang Sun - end

        return [firing_values, created_new_pc, current_PC]

    def compute_firing_values(self, gc_modules, virtual=False, axis=None, plot=False):
        if gc_modules[0].CUDA:

            s_vectors = np.empty((len(gc_modules), gc_modules[0].sheetDimension[0]*gc_modules[0].sheetDimension[1]))
            for m, gc in enumerate(gc_modules):
                s_vectors[m] = gc.get_s(virtual= virtual).flatten()
            firing_values = []
            for i, pc in enumerate(self.place_cells):
                if axis is not None:
                    plot = plot if i == 0 else False  # linear lookahead debugging plotting
                    firing = pc.compute_firing_2x(s_vectors, axis, plot=plot)  # firing along axis
                else:
                    firing = pc.compute_firing(s_vectors)  # overall firing

                firing_values.append(firing)
            return firing_values
        else:
            s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
            for m, gc in enumerate(gc_modules):
                if virtual:
                    s_vectors[m] = gc.s_virtual
                else:
                    s_vectors[m] = gc.s

            firing_values = []
            for i, pc in enumerate(self.place_cells):
                if axis is not None:
                    plot = plot if i == 0 else False  # linear lookahead debugging plotting
                    firing = pc.compute_firing_2x(s_vectors, axis, plot=plot)  # firing along axis
                else:
                    firing = pc.compute_firing(s_vectors)  # overall firing

                firing_values.append(firing)
            return firing_values

        # Consolidate grid cell spiking vectors that we want to consider


    def save_pc_network(self, filename=""):
        gc_connections = []
        env_coordinates = []
        for pc in self.place_cells:
            gc_connections.append(pc.gc_connections)
            env_coordinates.append(pc.env_coordinates)

        directory = self.path
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        np.save(directory+"gc_connections" + filename + ".npy", gc_connections)
        np.save(directory+"env_coordinates" + filename + ".npy", env_coordinates)


###changes by Haoyang Sun-start

class BlockCell:
    """Class to keep track of an individual Block Cell"""
    def __init__(self, gc_connections):
        self.gc_connections = gc_connections  # Connection matrix to grid cells of all modules; has form (n^2 x M)
        self.env_coordinates = None  # Save x and y coordinate at moment of creation

        self.plotted_found = [False, False]  # Was used for debug plotting, of linear lookahead

    def compute_firing(self, s_vectors):
        """Computes firing value based on current grid cell spiking"""
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # determine where connection exist to grid cells
        filtered = np.multiply(gc_connections, s_vectors)  # filter current grid cell spiking, by connections
        modules_firing = np.sum(filtered, axis=1) / np.sum(s_vectors, axis=1)  # for each module determine pc firing
        firing = np.average(modules_firing)  # compute overall pc firing by summing averaging over modules
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

    def __init__(self):
        self.block_cells = []  # array of block cells(all isolated)
        self.threshold = 0.85
    def create_new_bc(self, gc_modules):
        # Consolidate grid cell spiking vectors to matrix of size n^2 x M
        s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
        for m, gc in enumerate(gc_modules):
            s_vectors[m] = gc.s
        weights = compute_weights(s_vectors)
        bc = BlockCell(weights)
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
        """Keeps track of current grid cell firing"""
        gc_modules = gc_network.gc_modules
        ##get the obstacle datas from the environment module
        dire_vectors = env.get_blocked_directions()
        gc_network.reset_s_virtual()
        created_new_bc = False
        for n, vector in enumerate(dire_vectors):
            vector = np.multiply(vector,[0.5,0.5])
            gc_network.track_movement(vector, virtual=True, dt_alternative=2)
            firing_values = self.compute_firing_values(gc_network.gc_modules, virtual=True)
            # create a new block cell if there is no block cell present there
            if max(firing_values) < self.threshold:
                self.create_new_bc(gc_modules)
                created_new_bc = True
            gc_network.reset_s_virtual()






        ##firt get the blocked directions from the environment module

        ###changes by Haoyang Sun - start
        current_PC = self.check_current_PC(gc_modules)
        if generate_new_PC == False:
            return [firing_values, False, current_PC]
        ###changes by Haoyang Sun - end

        # if len(firing_values) == 0 or np.max(firing_values) < 0.85 or reward_first_found:
        if current_PC == -1 or reward_first_found:
            # No place cell shows significant excitement
            # If the random number is above a threshold a new pc is created
            self.create_new_bc(gc_modules)
            firing_values.append(1)
            creation_str = "Created new place cell with idx: " + str(len(self.block_cells) - 1) \
                           + " | Highest alternative spiking value: " + str(np.max(firing_values))
            # print(creation_str)
            # if reward_first_found:
            #     print("Found the goal and created place cell")
            created_new_pc = True
            ###changes by Haoyang Sun - start
            current_PC = len(self.place_cells) - 1
        ###changes by Haoyang Sun - end

        return [firing_values, created_new_pc, current_PC]

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

    def save_pc_network(self, filename=""):
        gc_connections = []
        env_coordinates = []
        for pc in self.place_cells:
            gc_connections.append(pc.gc_connections)
            env_coordinates.append(pc.env_coordinates)

        directory = "data/pc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/pc_model/gc_connections" + filename + ".npy", gc_connections)
        np.save("data/pc_model/env_coordinates" + filename + ".npy", env_coordinates)

###changes by Haoyang Sun-end



