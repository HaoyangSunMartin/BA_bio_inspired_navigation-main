from plotting.plotThesis import *
import os
import numpy as np

SPIKE_FIRING = 0.85

class PlaceCell_SIMPLE:
    # place cell is defined by an XY_coordinate and a firing function related to the distance
    """Class to keep track of an individual Place Cell"""
    def __init__(self, coordinate, thres=5.0, env_coordinates=None):
        self.position = coordinate
        self.thres = thres
        self.env_coordinates = env_coordinates


    def compute_firing(self, s_vectors):
        """Computes firing value based on the distance and a firing function"""
        dis = np.linalg.norm(self.position - s_vectors)

        return max(min(1.0, 1.0 - (dis/self.thres)), 0.0)

    def compute_firing_2x(self, s_vectors, axis, plot=False):
        """Computes firing projected on one axis, based on current grid cell spiking"""
        dis = np.abs(self.position[axis] - s_vectors[axis])

        return max(min(1.0, 1.0 - (dis/self.thres)), 0.0)


class PlaceCellNetwork_SIMPLE:
    """A PlaceCellNetwork holds information about all Place Cells"""
    def __init__(self, thres= 5.0, from_data=False):
        self.place_cells = []  # array of place cells
        self.thres = thres
        self.visited = []

        if from_data:
            pc_positions = np.load("data/pc_model/pc_positions.npy")
            pc_thresholds = np.load("data/pc_model/pc_thresholds.npy")
            env_coordinates = np.load("data/pc_model/env_coordinates.npy")

            for idx, position in enumerate(pc_positions):
                pc = PlaceCell_SIMPLE(position, thres=pc_thresholds[idx], env_coordinates=env_coordinates[idx])
                self.place_cells.append(pc)


    def create_new_pc(self, position):
        # Consolidate grid cell spiking vectors to matrix of size n^2 x M
        pc = PlaceCell_SIMPLE(position, thres=self.thres)
        self.place_cells.append(pc)

    ###changes by Haoyang Sun - start
    def check_current_PC(self, position):

        firing_values = self.compute_firing_values(position)
        if len(firing_values) == 0:
            return -1


        max_firing = max(firing_values)
        idx = firing_values.index(max_firing)
        if max_firing < 0.85:
            return -1
        else:
            return idx
    ###changes by Haoyang Sun - end

    def track_movement(self, position, reward_first_found, generate_new_PC = False):
        """Keeps track of current grid cell firing"""

        firing_values = self.compute_firing_values(position)

        for idx, firing in enumerate(firing_values):
            if firing > 0.85 and (idx not in self.visited):
                self.visited.append(idx)

        created_new_pc = False
        ###changes by Haoyang Sun - start
        current_PC = self.check_current_PC(position)
        if generate_new_PC == False:
            return [firing_values, False, current_PC]
        ###changes by Haoyang Sun - end



        #if len(firing_values) == 0 or np.max(firing_values) < 0.85 or reward_first_found:
        if current_PC == -1 or reward_first_found:
            # No place cell shows significant excitement
            # If the random number is above a threshold a new pc is created
            self.create_new_pc(position)
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

    def compute_firing_values(self, position, axis=None, plot=False):

        firing_values = []
        for i, pc in enumerate(self.place_cells):
            firing = pc.compute_firing(position)
            firing_values.append(firing)
        return firing_values




    def save_pc_network(self, filename=""):
        pc_positions = []
        pc_thresholds = []
        env_coordinates = []

        for pc in self.place_cells:
            pc_positions.append(pc.position)
            pc_thresholds.append(pc.thres)
            env_coordinates.append(pc.env_coordinates)

        directory = "data/pc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/pc_model/pc_positions" + filename + ".npy", pc_positions)
        np.save("data/pc_model/pc_thresholds" + filename + ".npy", pc_thresholds)
        np.save("data/pc_model/env_coordinates" + filename + ".npy", env_coordinates)

