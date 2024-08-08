from system.bio_model.gridcellModel import GridCellNetwork
from plotting.plotThesis import plot_grid_cell_modules
import numpy as np

dt = 1e-2

# initialize Grid Cell Model
M = 6  # number of modules
n = 40  # size of sheet -> nr of neurons is squared
gmin = 0.2  # maximum arena size, 0.5 -> ~10m | 0.05 -> ~105m
gmax = 2.4  # determines resolution, dont pick to high (>6 at speed = 2)
# if gc modules are created from data n and M are overwritten
gc_network = GridCellNetwork(n, M, dt, gmin, gmax=gmax, from_data=True)


# pc_network = PlaceCellNetwork()
# pc_network.create_new_pc(gc_network.gc_modules)
#
# firing0 = pc_network.compute_firing_values(gc_network.gc_modules)
# print("Initial firing:, ", firing0)
#
speed = np.ones((3000, 2)) * 0.5
pause = np.zeros((1000, 2))

xy_speeds = np.concatenate((speed, pause, - speed, pause))

for idx, xy_speed in enumerate(xy_speeds):
    if idx % 1000 == 0:
        # plot_grid_cell_modules(gc_network.gc_modules, idx, plot_target=True, plot_matches=False)
        plot_grid_cell_modules(gc_network.gc_modules, idx, plot_target=True, plot_matches=True)
    gc_network.track_movement(xy_speed)
#
# plot_grid_cell_modules(gc_network.gc_modules, 8000, plot_target=True, plot_matches=False)
# plot_grid_cell_modules(gc_network.gc_modules, 8000, plot_target=True, plot_matches=True)
# firing1 = pc_network.compute_firing_values(gc_network.gc_modules)
# print("Final firing: ", firing1)
