import math

from plotting.plotThesis import plot_grid_cell_modules, plot_3D_sheets
import numpy as np
import os
from numba import cuda
import numba.cuda
from numba import jit
###?
"""
here is the CUDA Implementation of the grid cell module:
"""

@cuda.jit
def calculate_distance_CUDA(dimension, dis_matrix, de):
    #xx,yy is the absolute distance of the two dimensions in the neuron sheet
    xx,yy = dimension
    #make the distance between the edge neurons be one in the wrap around fashion
    xx= xx+1
    yy= yy+1

    x, y = cuda.grid(2)
    if x < dis_matrix.shape[0] and y< dis_matrix.shape[1]:
        #this thread corresponds to the calculation between neuron A(ax,ay) and B(bx,by)
        ax = x % xx
        ay = math.floor(x / xx)
        bx = y % xx
        by =math.floor(y / xx)
        #get the preferred heading:
        p = 2*(ay % 2)+ ax % 2
        #tune the coordinate according to the preferred direction:
        # [[-1, 0], [0, 1], [0, -1], [1, 0]]  # [W, N, S, E]
        if p ==0:
            ax = ax-de
        if p ==1:
            ay = ay+de
        if p ==2:
            ay = ay-de
        if p ==3:
            ax = ax+de
        #calculate the absolute distance between A and B in a Wrap Around manner
        dx = abs(ax - bx)
        if dx > (xx * 0.5):
            dx = xx - dx
        dy = abs(ay - by)
        if dy > (yy * 0.5):
            dy = yy - dy
        #write the distance to the matrix
        dis_matrix[x,y] = (dx*dx) + (dy*dy)

@cuda.jit
def rec_d_CUDA(dis_matrix, r, lamda):
    x,y = cuda.grid(2)
    beta = 3 / (lamda*lamda)
    if x < dis_matrix.shape[0] and y < dis_matrix.shape[1]:
        dis_matrix[x,y] = np.exp(-1*r*dis_matrix[x,y]) - np.exp(-1*beta*dis_matrix[x,y])

@cuda.jit
def update_GC_CUDA(w_matrix, s_vector, movement, gm ,de):
    #this function uses the weight matrix stored in the constant memory of the GPU
    #and the current s_vector and the movement vector to calculate the next s_vector of the GC module
    x, y  = cuda.grid(2)
    if x < s_vector.shape[0] and y < s_vector.shape[1]:
        #the thread [x,y] is responsible for updating the neuron A[x,y]
        gidx = x + y*s_vector.shape[0]
        tmp = 0
        #get the preferred direction
        p = 2 * (y % 2) + x % 2
        dire = np.array([0,0])
        # tune the coordinate according to the preferred direction:
        # [[-1, 0], [0, 1], [0, -1], [1, 0]]  # [W, N, S, E]
        if p == 0:
            dire = np.array([-1*de, 0])
        if p == 1:
            dire = np.array([0, de])
        if p == 2:
            dire = np.array([0, -1*de])
        if p == 3:
            dire = np.array([de, 0])
        for idx in range(w_matrix.shape[1]):
            xx = idx % s_vector.shape[0]
            yy = math.floor(idx / s_vector.shape[0])
            tmp = tmp + s_vector[xx, yy] * w_matrix[gidx, idx]
        B = 1+ 0.10315*gm*np.dot(dire, movement)
        tmp = tmp + B
        tmp = max(0, tmp)
        cuda.syncthreads()
        s_vector[x,y] = tmp














###?
# Grid Cell model is based on Edvardsen 2015. Please refer to the thesis or the paper for detailed explanations


def rec_d(d):
    """Recurrent connectivity profile used for calculating connection weight between neurons"""
    lam = 15  # determines periodicity of pattern (lambda ~ #neurons between)
    beta = 3 / (lam ** 2)
    gamma = 1.05 * beta

    weight = np.exp(-gamma * (d ** 2)) - np.exp(-beta * (d ** 2))
    return weight


# Computes distance between
def compute_ds(x1, x2):
    """Calculates distance in along one axis between x1 and x2
    x1: vector of x coordinates (eg. [0, 0, 0, 1, 1, 1, 2, 2, 2]) of size n^2
    x2: vector of other x coordinates (eg. [1, 1, 1, 2, 1, 0, 3, 1, 2]) of size n^2
    returns dx from each neuron to each neuron, matrix of size n^2 x n^2
    """
    n = int(np.sqrt(len(x1)))  # size of grid cell sheet (width and height)
    x1 = np.tile(x1, (n**2, 1))  # tile vector to fit size n^2 x n^2
    x2 = np.transpose(np.tile(x2, (n**2, 1)))  # tile vector to fit size n^2 x n^2, but transpose
    dx1 = np.abs(x2 - x1)  # calculate distance from each neuron to each neuron
    dx2 = n - dx1  # as edges of grid cell sheet are connected dx < n -> calculate other way
    dx = np.min([dx1, dx2], axis=0)  # choose shortest path (left or right)
    return dx  # return matrix of size n^2 x n^2, representing shortest distance along 1 axis


def compute_gm(m, M, gmin, gmax=None):
    """Calculates velocity gain factor g_m for all modules according to formula in thesis"""

    if gmax is None:
        # If only 1 boundary is provided, we let the gain factor increase linearly (Edvardsen 2017)
        gm = gmin * 1.5 ** m
    else:
        # Otherwise we make sure the gain factors are properly spaced between the boundaries (Edvardsen 2015)
        if M != 1:
            R = np.power(gmax / gmin, 1 / (M - 1))
        else:
            R = 1
        gm = gmin * np.power(R, m)

    return gm


def implicit_euler(s0, w, b, tau, dt):
    """Solve the grid cell spiking equation with implicit euler for one time step of size dt"""
    f = np.maximum(0, np.tensordot(s0, w, axes=1) + b)
    s = (s0 + f * dt / tau) / (1 + dt / tau)
    return s


# Not used, but defined grid cell spiking equation to be solved with built in numeric solver
def ds_dt(t, s, w, b, tau):
    f = np.maximum(0, np.tensordot(s, w, axes=1) + b)
    return (f - s) / tau





class GridCellModule:
    """One GridCellModule holds the information of a sheet of n x n neurons"""
    def __init__(self, n, gm, dt, tuned_direction='x',data=None):
        self.CUDA = False
        self.n = n  # Grid Cell sheet size (height and width)
        self.gm = gm  # velocity gain factor

        array_length = n**2

        # connection weight matrix from each to each neuron
        self.w = np.random.random_sample((array_length, array_length))

        # self.h = np.random.random_sample((array_length, 2))
        #self.s = np.random.rand(array_length) * 10**-4  # firing vector of size (n^2 x 1); random firing at beginning
        self.s = np.zeros(array_length) * 10 ** -4
        self.t = self.s  # target grid cell firing (of goal or home-base)
        self.s_virtual = self.s  # used for linear lookahead to preplay trajectories, without actually moving
        self.dt = dt  # time step size

        self.s_video_array = []
        self.tuned_direction = tuned_direction
        # If we are not loading grid cell data we have to calculate grid cell sheet weights
        if data is None:
            W = [-1, 0]
            N = [0, 1]
            S = [0, -1]
            E = [1, 0]
            # Refer to thesis for concept of grid cell sheet and how weights are computed
            if tuned_direction == 'x':
                headings = [W, N, S, E]  # [W, N, S, E]
            else:
                headings = [W, S, N, E]  # [E, S, N, W]
            grid = np.indices((n, n))  # grid function to create x and y vectors
            x = np.concatenate(grid[1])  # x vector of form eg. [0, 0, 0, 1, 1, 1, 2, 2, 2]
            y = np.concatenate(grid[0])  # y vector of form eg. [0, 1, 2, 0, 1, 2, 0, 1, 2]
            index = 2 * np.mod(y, 2) + np.mod(x, 2)  # refer to thesis for explanation of formula
            index.astype(int)
            self.h = np.take(headings, index, axis=0)  # pick preferred heading direction for each neuron
            """
            for 4x4 grid the heading would be:
              3 S  E  S  E
            y 2 W  N  W  N
              1 S  E  S  E
              0 W  N  W  N
                0  1  2  3
                   x
            """

            x_tuned = np.subtract(x, self.h[:, 0])  # tune x vector according to preferred heading direction
            y_tuned = np.subtract(y, self.h[:, 1])  # tune y vector according to preferred heading direction

            dx = compute_ds(x_tuned, x)  # compute shortest x distance between each pair of neurons (i - e_i, j)
            dy = compute_ds(y_tuned, y)  # compute shortest y distance between each pair of neurons (i - e_i, j)
            d = np.linalg.norm([dx, dy], axis=0)  # compute shortest overall distance between each pair of neurons
            self.w = rec_d(d)  # apply recurrent connectivity profile to get weights
        else:
            self.w = data["w"]
            self.h = data["h"]
    def get_s(self, virtual = False):
        if virtual:
            return self.s_virtual
        return self.s
    def update_s(self, v, virtual=False, dt_alternative=None):
        """Updates grid cell spiking from one to next time step"""

        tau = 1e-1  # defined by model
        alpha = 0.10315  # defined by model

        g = self.gm

        s0 = self.s_virtual if virtual else self.s  # virtual or actual mode
        dt = self.dt if dt_alternative is None else dt_alternative  # determine wanted time step size

        b = 1 + g * alpha * np.tensordot(self.h, v, axes=1)  # calculate b according to formula

        s = self.s
        if dt_alternative is None:
            # apply implicit euler once to update spiking
            s = implicit_euler(s0, self.w, b, tau, dt)
        else:
            # Alternative approach to use built in solver to calculate bigger time steps at once, large computation time
            # Because Implicit euler is unstable for large dt
            # sol = solve_ivp(ds_dt, (0, dt_alternative), s0, t_eval=[dt_alternative], args=(self.w, b, tau))
            # s = sol.y[:, 0]

            # It is faster to just apply the implicit euler several times until targeted time step is reached
            for n in range(int(dt_alternative/self.dt)):
                s0 = implicit_euler(s0, self.w, b, tau, self.dt)
            s = s0

        if virtual:
            self.s_virtual = s  # updates spiking value
            self.s_video_array.append(self.s_virtual)  # save for lookahead video
        else:
            s = implicit_euler(s0, self.w, b, tau, dt)  # this step might actually not be necessary, pls investigate
            self.s = s  # updates spiking value


class GridCellNetwork:
    """GridCellNetwork holds all Grid Cell Modules"""
    def __init__(self, n, M, dt, gmin, gmax=None, from_data=False, randomized_init =False):
        self.CUDA = False
        self.gc_modules = []  # array holding objects GridCellModule
        self.dt = dt
        self.random_init = randomized_init
        self.tuned_vector = []
        self.gm_vector = []
        self.n = n
        if not from_data:
            # Create new GridCellModules
            for i, m in enumerate(range(M)):

                gm = compute_gm(m, M, gmin, gmax)
                self.gm_vector.append(gm)
                if i % 2 == 0:
                    direction = 'x'
                else:
                    direction = 'y'
                gc = GridCellModule(n, gm, dt, tuned_direction=direction)
                self.gc_modules.append(gc)
                print("Created GC module with gm", gc.gm, " tuned to the direction ", direction )

            for i in self.gc_modules:
                self.tuned_vector.append(i.tuned_direction)
            self.save_gc_model()
            nr_steps_init = 1000
            self.initialize_network(nr_steps_init, "s_vectors_initialized.npy")

        else:
            # Load previous data
            w_vectors = np.load("data/LINEAR/gc_model/w_vectors.npy")
            h_vectors = np.load("data/LINEAR/gc_model/h_vectors.npy")
            gm_values = np.load("data/LINEAR/gc_model/gm_values.npy")

            self.tuned_vector = np.load("data/LINEAR/gc_model/tuned_direction.npy")
            self.gm_vector = gm_values
            n = int(np.sqrt(len(w_vectors[0][0])))
            for m, gm in enumerate(gm_values):
                gc = GridCellModule(n, gm, dt, data={"w": w_vectors[m], "h": h_vectors[m]},tuned_direction=self.tuned_vector[m])
                self.gc_modules.append(gc)
                print("Loaded GC module with gm", gc.gm, " tuned to direction ", self.tuned_vector[m])

            self.load_initialized_network("s_vectors_initialized.npy")

        self.set_current_as_target_state()  # by default home-base is set as goal vector





    def track_movement(self, xy_speed, virtual=False, dt_alternative=None):
        """For each grid cell module update spiking"""
        for gc in self.gc_modules:
            gc.update_s(xy_speed, virtual=virtual, dt_alternative=dt_alternative)

    def initialize_network(self, nr_steps, filename):
        """For each grid cell module initialize spiking"""
        if self.random_init:
            xy_speed = [0, 0]
            for i in range(nr_steps):
                if np.random.random() > 0.95:
                    # Apply a small velocity vector in some cases to ensure that peaks form
                    xy_speed = np.random.rand(2) * 0.2
                self.track_movement(xy_speed)
                #if i % 499 == 0:
                    #print("Currently at Timestep:", i)
                    # plot_grid_cell_modules(self.gc_modules, i)
                    #plot_3D_sheets(self.gc_modules, i)
            print("Finished Initialization of nr_steps:", nr_steps)
        else:

            xy_speed_array = [np.array([0.5, 0.0]), np.array([0.0, 0.5]),
                              np.array([-0.5, 0.0]), np.array([0.0, -0.5])]
            nr = math.floor(nr_steps / 4)
            for i in range(nr_steps):
                # if i % nr == 0:
                # print("Currently at Timestep:", i)
                # plot_3D_sheets(self.gc_modules, i)

                # print("Currently at Timestep:", i)
                xy_speed = xy_speed_array[math.floor(i / nr)]
                self.track_movement(xy_speed)
            for i in range(nr):
                xy_speed = np.array([0.2, 0.2], dtype=np.double)
                self.track_movement(xy_speed)
            print("Finished Initialization of nr_steps:", nr_steps)

        #plot_grid_cell_modules(self.gc_modules, nr_steps)
        #plot_3D_sheets(self.gc_modules, nr_steps)

        self.save_gc_spiking(filename)

    def load_initialized_network(self, filename):
        s_vectors = np.load("data/LINEAR/gc_model/" + filename)
        for m, gc in enumerate(self.gc_modules):
            gc.s = s_vectors[m]
        # plot_grid_cell_modules(self.gc_modules, "final")
        # plot_3D_sheets(self.gc_modules, "final")

    def save_gc_model(self):
        w_vectors = []
        h_vectors = []
        gm_values = []
        tuned_vector = []
        for gc in self.gc_modules:
            w_vectors.append(gc.w)
            h_vectors.append(gc.h)
            gm_values.append(gc.gm)
            tuned_vector.append(gc.tuned_direction)

        directory = "data/LINEAR/gc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/LINEAR/gc_model/w_vectors.npy", w_vectors)
        np.save("data/LINEAR/gc_model/h_vectors.npy", h_vectors)
        np.save("data/LINEAR/gc_model/gm_values.npy", gm_values)
        np.save("data/LINEAR/gc_model/tuned_direction.npy", tuned_vector)

    def consolidate_gc_spiking(self, virtual=False):
        """Consolidate spiking in one matrix for saving"""
        s_vectors = np.zeros((len(self.gc_modules), len(self.gc_modules[0].s)))
        for idx, gc in enumerate(self.gc_modules):
            s = gc.s if not virtual else gc.s_virtual
            s_vectors[idx] = s
        return s_vectors

    def save_gc_spiking(self, filename):
        s_vectors = self.consolidate_gc_spiking()

        directory = "data/LINEAR/gc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/LINEAR/gc_model/"+filename, s_vectors)

    def set_current_as_target_state(self):
        for m, gc in enumerate(self.gc_modules):
            gc.t = np.copy(gc.s)

    def set_as_target_state(self, gc_connections):
        for m, gc in enumerate(self.gc_modules):
            gc.t = gc_connections[m]
        print("Set new target state")

    def reset_s_virtual(self):
        for m, gc in enumerate(self.gc_modules):
            gc.s_virtual = np.copy(gc.s)
            gc.s_video_array.clear()

    def set_filename_as_target_state(self, filename):
        t_vectors = np.load("data/LINEAR/gc_model/" + filename)
        for m, gc in enumerate(self.gc_modules):
            gc.t = t_vectors[m]
        print("Set loaded data as new target state:", filename)
