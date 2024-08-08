import math

from plotting.plotThesis import plot_grid_cell_modules, plot_3D_sheets
import numpy as np
import os
from numba import cuda
import numba.cuda
from numba import jit
###?

class GridCellModule:
    """One GridCellModule holds the information of a sheet of n x n neurons"""
    def __init__(self, n, gm, dt, data=None):
        self.current = np.array([0.0,0.0])#xy coordinate
        self.dt = dt
        self.virtual = np.copy(self.current)

    def get_s(self, virtual=False):
        if virtual:
            return np.copy(self.virtual)
        else:
            return np.copy(self.current)

    def update_s(self, v, virtual=False, dt_alternative=None):
        t = self.dt if dt_alternative is None else dt_alternative
        x=v[0]
        y=v[1]

        if virtual:
            self.virtual = self.virtual + np.array([t*x,t*y])
        else:
            self.current = self.current + np.array([t*x,t*y])

    def reset_virtual(self):
        self.virtual = np.copy(self.current)

class GridCellNetwork:
    """GridCellNetwork holds all Grid Cell Modules"""
    def __init__(self, n, M, dt, gmin, gmax=None, from_data=False):
       self.gc = GridCellModule(n,gmin, dt, from_data)

    def track_movement(self, xy_speed, virtual=False, dt_alternative=None):
        """For each grid cell module update spiking"""
        self.gc.update_s(xy_speed, virtual, dt_alternative)

    def get_position(self, virtual=False):
        return self.gc.get_s(virtual)


    def initialize_network(self, nr_steps, filename):
        """For each grid cell module initialize spiking"""
        tmp = "this is actually not needed if you really think about it"

    def load_initialized_network(self, filename):
        tmp = "this is actually not needed if you really think about it"

    def save_gc_model(self):
        tmp = "this is actually not needed if you really think about it"

    def consolidate_gc_spiking(self, virtual=False):
        """Consolidate spiking in one matrix for saving"""

        return self.gc.get_s(virtual)

    def save_gc_spiking(self, filename):
        tmp = "this is actually not needed if you really think about it"

    def set_current_as_target_state(self):
        tmp = "this is actually not needed if you really think about it"

    def set_as_target_state(self, gc_connections):
        tmp = "this is actually not needed if you really think about it"

    def reset_s_virtual(self):
        self.gc.reset_virtual()

    def set_filename_as_target_state(self, filename):
        tmp = "this is actually not needed if you really think about it"
