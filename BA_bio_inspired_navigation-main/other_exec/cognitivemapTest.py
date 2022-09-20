from system.bio_model.gridcellModel import GridCellNetwork
from system.bio_model.placecellModel import PlaceCellNetwork
from system.bio_model.cognitivemapModel import CognitiveMapNetwork

from system.controller.pybulletEnv import PybulletEnvironment

from plotting.plotThesis import plot_vector_navigation_error

import numpy as np


dt = 1e-2

# initialize Pybullet Environment
visualize = False
env_model = "linear_sunburst"  # "plane" for default, "single_line_traversal", "linear_sunburst"
vector_model = "linear_lookahead"  # "linear_lookahead" for default, "phase_offset_detector", "spike_detection"

# initialize Grid Cell Model
M = 6  # number of modules
n = 40  # size of sheet -> nr of neurons is squared
gmin = 0.2  # maximum arena size, 0.5 -> ~10m | 0.05 -> ~105m
gmax = 2.4  # determines resolution, dont pick to high (>6 at speed = 2)
# if gc modules are created from data n and M are overwritten
gc_network = GridCellNetwork(n, M, dt, gmin, gmax=gmax, from_data=True)

env = PybulletEnvironment(visualize, env_model, dt)

from_data = True

# initialize Place Cell Model
pc_network = PlaceCellNetwork(from_data=from_data)
# initialize Cognitive Map Model
cognitive_map = CognitiveMapNetwork(dt, from_data=from_data)

# goal_vectors = []
# for goal_pc_idx in range(len(pc_network.place_cells)):
#     gc_network.set_as_target_state(pc_network.place_cells[goal_pc_idx].gc_connections)
#     goal_vector = perform_look_ahead_2x(gc_network, pc_network, cognitive_map, env, goal_pc_idx=goal_pc_idx,
#                                         video=False, plotting=False)
#     goal_vectors.append(goal_vector)
#
# print(goal_vectors)
# np.save("experiments/cognitive_map/vectors_array", np.array(goal_vectors))

vectors_array = np.load("experiments/cognitive_map/vectors_array.npy")

# cognitive_map_plot(pc_network, cognitive_map, vectors_array=vectors_array)

error_array = np.empty((len(vectors_array)))
for idx, vec in enumerate(vectors_array):
    origin = np.array([5.5, 0.5])
    real_vec = pc_network.place_cells[idx].env_coordinates - origin
    error_array[idx] = np.linalg.norm(vec - real_vec)

plot_vector_navigation_error(error_array)
