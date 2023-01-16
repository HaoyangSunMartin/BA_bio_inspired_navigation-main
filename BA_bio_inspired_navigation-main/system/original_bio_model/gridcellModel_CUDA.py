import math
from plotting.plotThesis import plot_grid_cell_modules, plot_3D_sheets
import numpy as np
import os
from numba import cuda

###?
"""
here is the CUDA Implementation of the grid cell module:
"""


@cuda.jit
def calculate_distance_CUDA(dimension, dis_matrix, de):
    # xx,yy is the absolute distance of the two dimensions in the neuron sheet
    xx, yy = dimension
    # make the distance between the edge neurons be one in the wrap around fashion
    xx = xx
    yy = yy

    x, y = cuda.grid(2)
    if x < dis_matrix.shape[0] and y < dis_matrix.shape[1]:
        # this thread corresponds to the calculation between neuron A(ax,ay) and B(bx,by)
        ax = x % xx
        ay = math.floor(x / xx)
        bx = y % xx
        by = math.floor(y / xx)
        # get the preferred heading:
        p = 2 * (ay % 2) + ax % 2
        # tune the coordinate according to the preferred direction:
        # [[-1, 0], [0, 1], [0, -1], [1, 0]]  # [W, N, S, E]
        if p == 0:
            ax = ax + de
        if p == 1:
            ay = ay - de
        if p == 2:
            ay = ay + de
        if p == 3:
            ax = ax - de
        # calculate the absolute distance between A and B in a Wrap Around manner
        dx = abs(ax - bx)
        # if dx > (xx * 0.5):
        #    dx = xx - dx
        dx = min(dx, xx - dx)
        dy = abs(ay - by)
        # if dy > (yy * 0.5):
        #    dy = yy - dy
        dy = min(dy, yy - dy)
        # write the distance to the matrix
        dis_matrix[x, y] = (dx * dx) + (dy * dy)


@cuda.jit
def rec_d_CUDA(dis_matrix, r, lamda):
    x, y = cuda.grid(2)
    beta = 3 / (lamda * lamda)
    if x < dis_matrix.shape[0] and y < dis_matrix.shape[1]:
        dis_matrix[x, y] = math.exp(-1 * r * beta * dis_matrix[x, y]) - math.exp(-1 * beta * dis_matrix[x, y])


@cuda.jit
def update_GC_CUDA(w_matrix, s_vector, x_movement, y_movement, gm, W, N, S, E, time_parameters):
    # this function uses the weight matrix stored in the constant memory of the GPU
    # and the current s_vector and the movement vector to calculate the next s_vector of the GC module
    x, y = cuda.grid(2)
    tau = time_parameters[0]
    dt = time_parameters[1]

    if x < s_vector.shape[0] and y < s_vector.shape[1]:
        # the thread [x,y] is responsible for updating the neuron A[x,y]
        gidx = x + y * s_vector.shape[0]  # get the correspondent index in the weight Matrix gidx
        tmp = 0
        # get the preferred direction
        p = 2 * (y % 2) + x % 2

        # tune the coordinate according to the preferred direction:
        # [[-1, 0], [0, 1], [0, -1], [1, 0]]  # [W, N, S, E]
        if p == 0:
            dire = W
        if p == 1:
            dire = N
        if p == 2:
            dire = S
        if p == 3:
            dire = E
        # do a matrix multiplication
        for idx in range(w_matrix.shape[1]):
            # translate the index in weight matrix to the index in s_vector matrix idx->[xx,yy]
            xx = idx % s_vector.shape[0]
            yy = int(math.floor(idx / s_vector.shape[0]))

            tmp = tmp + s_vector[xx, yy] * w_matrix[gidx, idx]
        tmp1 = dire[0] * x_movement + dire[1] * y_movement
        tmp = tmp + 1.0 + 0.10315 * gm * tmp1
        tmp = max(0, tmp)
        tmp = (s_vector[x, y] + (tmp * dt / tau)) / (1 + (dt / tau))
        cuda.syncthreads()
        s_vector[x, y] = tmp

@cuda.jit
def update_GC_CUDA_Alternative(w_matrix, s_vector, x_movement, y_movement, gm, W, N, S, E, time_parameters):
    # this function uses the weight matrix stored in the constant memory of the GPU
    # and the current s_vector and the movement vector to calculate the next s_vector of the GC module
    x, y = cuda.grid(2)
    tau = time_parameters[0]
    dt = time_parameters[2]

    if x < s_vector.shape[0] and y < s_vector.shape[1]:
        # the thread [x,y] is responsible for updating the neuron A[x,y]
        gidx = x + y * s_vector.shape[0]  # get the correspondent index in the weight Matrix gidx
        tmp = 0
        # get the preferred direction
        p = 2 * (y % 2) + x % 2

        # tune the coordinate according to the preferred direction:
        # [[-1, 0], [0, 1], [0, -1], [1, 0]]  # [W, N, S, E]
        if p == 0:
            dire = W
        if p == 1:
            dire = N
        if p == 2:
            dire = S
        if p == 3:
            dire = E
        # do a matrix multiplication
        for idx in range(w_matrix.shape[1]):
            # translate the index in weight matrix to the index in s_vector matrix idx->[xx,yy]
            xx = idx % s_vector.shape[0]
            yy = int(math.floor(idx / s_vector.shape[0]))

            tmp = tmp + s_vector[xx, yy] * w_matrix[gidx, idx]
        tmp = tmp + (1 + 0.10315 * gm * (dire[0] * x_movement + dire[1] * y_movement))
        tmp = max(0, tmp)
        tmp = (s_vector[x, y] + (tmp * dt / tau)) / (1 + (dt / tau))
        cuda.syncthreads()
        s_vector[x, y] = tmp




class GridCellModule_CUDA:
    def __init__(self, sheetDimension, dt, gm=0.2, de=1.0, r=1.05, lamda=15, tau=1e-1):
        self.CUDA = True
        self.sheetDimension = sheetDimension
        self.weightMatrixDimension = (
        self.sheetDimension[0] * self.sheetDimension[1], self.sheetDimension[0] * self.sheetDimension[1])
        self.h_weightMatrix = np.ones(self.weightMatrixDimension)
        self.d_weightMatrix = cuda.to_device(self.h_weightMatrix)
        #host vector for back-up(for instance LLA) of the actual gridcell Module(this should not be updated frequently)
        self.h_sVector = np.zeros(self.sheetDimension)
        #device vector is responsible for states update computation in the GPU
        self.d_sVector = cuda.to_device(self.h_sVector)
        #host vector for output of the s_Vector,
        #the GPU only prints to this array and should never read from this arrray
        self.output_sVector = np.copy(self.h_sVector)
        #target state vector
        self.t = np.copy(self.h_sVector)
        #this boolean defines if the model is doing virtual calculation or the environment calculation
        self.virual = False

        self.de = de
        self.gm = gm
        self.r = r
        self.lamda = lamda
        self.W = cuda.to_device(np.array([-1 * de, 0]))
        self.N = cuda.to_device(np.array([0, de]))
        self.S = cuda.to_device(np.array([0, -1 * de]))
        self.E = cuda.to_device(np.array([de, 0]))
        self.tau = tau
        self.dt = dt

        self.time_parameters = cuda.to_device(np.array([self.tau, self.dt, self.dt * 10]))

        ##this block is added to be compatable with the linear version
        headings = [[-1, 0], [0, 1], [0, -1], [1, 0]]  # [W, N, S, E]
        grid = np.indices(self.sheetDimension)  # grid function to create x and y vectors
        x = np.concatenate(grid[1])  # x vector of form eg. [0, 0, 0, 1, 1, 1, 2, 2, 2]
        y = np.concatenate(grid[0])  # y vector of form eg. [0, 1, 2, 0, 1, 2, 0, 1, 2]
        index = 2 * np.mod(y, 2) + np.mod(x, 2)  # refer to thesis for explanation of formula
        index.astype(int)
        self.h = np.take(headings, index, axis=0)  # pick preferred heading direction for each neuron
        ##

        # declare the CUDA Structure for weightMatrix Manipulation
        self.threadsPerBlock = (8, 8)
        blocksPerGrid_x = math.ceil(self.weightMatrixDimension[0] / self.threadsPerBlock[0])
        blocksPerGrid_y = math.ceil(self.weightMatrixDimension[1] / self.threadsPerBlock[1])
        self.blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
        # declare the CUDA structure for s_vectorMatrix Manipulation
        blocksPerGrid_sVector_x = math.ceil(self.sheetDimension[0] / self.threadsPerBlock[0])
        blocksPerGrid_sVector_y = math.ceil(self.sheetDimension[1] / self.threadsPerBlock[1])
        self.blocksPerGrid_sVector = (blocksPerGrid_sVector_x, blocksPerGrid_sVector_y)

    def initializeWeightMatrix(self, data = None):
        if data is None:
            calculate_distance_CUDA[self.blocksPerGrid, self.threadsPerBlock](self.sheetDimension, self.d_weightMatrix,
                                                                              self.de)
            rec_d_CUDA[self.blocksPerGrid, self.threadsPerBlock](self.d_weightMatrix, self.r, self.lamda)
        else:
            self.h_weightMatrix = np.reshape(data["w"], self.weightMatrixDimension)
            self.d_weightMatrix = cuda.to_device(self.h_weightMatrix)
            self.h = data["h"]

    def initialize_sVector(self, data):
        self.h_sVector = np.reshape(data, self.sheetDimension)
        self.output_sVector = np.copy(self.h_sVector)
        self.d_sVector = cuda.to_device(self.h_sVector)

    def update_s(self, xy_speed, virtual = False, dt_alternative = None):
        if self.virual==False and virtual==True:
            self.prepare_virtual()
        if self.virual==True and virtual == False:
            self.reset_s_virtual()
        #print("updating the gridCell module with gm: ", self.gm," with movement: ", xy_speed )
        if dt_alternative is None:

            update_GC_CUDA[self.blocksPerGrid_sVector, self.threadsPerBlock] \
                (self.d_weightMatrix, self.d_sVector, xy_speed[0], xy_speed[1], self.gm, self.W, self.N, self.S, self.E,
                 self.time_parameters)
        else:
            update_GC_CUDA_Alternative[self.blocksPerGrid_sVector, self.threadsPerBlock]\
                (self.d_weightMatrix, self.d_sVector, xy_speed[0], xy_speed[1], self.gm, self.W, self.N, self.S, self.E,
                 self.time_parameters)


    #this function should be called every time the model switches to virtual updates
    def prepare_virtual(self):
        self.h_sVector = self.d_sVector.copy_to_host()
        self.virual =True

    #this function ends the virtual update phase
    def reset_s_virtual(self):
        self.d_sVector = cuda.to_device(self.h_sVector)
        self.virual = False
    def get_s(self, virtual = False):
        if self.virual:
            if virtual:
                self.output_sVector = self.d_sVector.copy_to_host()
            else:
                self.output_sVector = np.copy(self.h_sVector)
        else:
            self.output_sVector = self.d_sVector.copy_to_host()

        return self.output_sVector



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


class GridCellNetwork_CUDA:
    """GridCellNetwork holds all Grid Cell Modules"""
    def __init__(self, n, M, dt, gmin, gmax=None, from_data=False):
        self.CUDA = True
        self.gc_modules = []  # array holding objects GridCellModule
        self.dt = dt

        if not from_data:
            # Create new GridCellModules
            for m in range(M):
                gm = compute_gm(m, M, gmin, gmax)
                gc = GridCellModule_CUDA((n,n),dt, gm= gm)
                self.gc_modules.append(gc)
                print("Created GC module with gm", gc.gm)
            self.save_gc_model()
            nr_steps_init = 1000
            self.initialize_network(nr_steps_init, "s_vectors_initialized.npy")

        else:
            # Load previous data
            w_vectors = np.load("data/gc_model/w_vectors.npy")
            h_vectors = np.load("data/gc_model/h_vectors.npy")
            gm_values = np.load("data/gc_model/gm_values.npy")
            print(w_vectors.shape)
            n = int(np.sqrt(np.sqrt(len(w_vectors[0]))))
            for m, gm in enumerate(gm_values):
                gc = GridCellModule_CUDA((n,n), gm = gm, dt = dt)
                gc.initializeWeightMatrix(data = {"w": w_vectors[m], "h": h_vectors[m]})
                self.gc_modules.append(gc)
                print("Loaded GC module with gm", gc.gm)

            self.load_initialized_network("s_vectors_initialized.npy")

        self.set_current_as_target_state()  # by default home-base is set as goal vector

    def track_movement(self, xy_speed, virtual=False, dt_alternative=None):
        """For each grid cell module update spiking"""
        for gc in self.gc_modules:

            gc.update_s(xy_speed, virtual=virtual, dt_alternative=dt_alternative)

    def initialize_network(self, nr_steps, filename):
        """For each grid cell module initialize spiking"""
        for gc in self.gc_modules:
            gc.initializeWeightMatrix()

        xy_speed_array = [np.array([0.5, 0.0], dtype=np.double), np.array([0.0, 0.5], dtype=np.double), np.array([-0.5, 0.0], dtype=np.double), np.array([0.0, -0.5], dtype=np.double)]
        nr = math.floor(nr_steps / 4)
        for i in range(nr_steps):
            #if i % nr == 0:
                #print("Currently at Timestep:", i)
                #plot_3D_sheets(self.gc_modules, i)

            #print("Currently at Timestep:", i)
            xy_speed = xy_speed_array[math.floor(i / nr)]
            self.track_movement(xy_speed)
        for i in range(nr):
            xy_speed = np.array([0.2, 0.2], dtype=np.double)
            self.track_movement(xy_speed)
        print("Finished Initialization of nr_steps:", nr_steps)
        #plot_grid_cell_modules(self.gc_modules, nr_steps)
        plot_3D_sheets(self.gc_modules, nr_steps)

        self.save_gc_spiking(filename)

    def load_initialized_network(self, filename):
        s_vectors = np.load("data/gc_model/" + filename)
        for m, gc in enumerate(self.gc_modules):
            gc.initialize_sVector(s_vectors[m])
        # plot_grid_cell_modules(self.gc_modules, "final")
        plot_3D_sheets(self.gc_modules, "final")

    def save_gc_model(self):
        w_vectors = []
        h_vectors = []
        gm_values = []
        for gc in self.gc_modules:

            w_vectors.append(np.concatenate(gc.h_weightMatrix))
            h_vectors.append(gc.h)
            gm_values.append(gc.gm)

        directory = "data/gc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/gc_model/w_vectors.npy", w_vectors)
        np.save("data/gc_model/h_vectors.npy", h_vectors)
        np.save("data/gc_model/gm_values.npy", gm_values)

    def consolidate_gc_spiking(self, virtual=False):
        """Consolidate spiking in one matrix for saving"""
        s_vectors = np.zeros((len(self.gc_modules), self.gc_modules[0].sheetDimension[0]*self.gc_modules[0].sheetDimension[1]))
        for idx, gc in enumerate(self.gc_modules):
            s = np.concatenate(gc.get_s(virtual))
            s_vectors[idx] = s
        return s_vectors

    def save_gc_spiking(self, filename):
        s_vectors = self.consolidate_gc_spiking()
        print("saving the s_vectors with shape: ", s_vectors.shape)

        directory = "data/gc_model/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("data/gc_model/"+filename, s_vectors)

    def set_current_as_target_state(self):
        for m, gc in enumerate(self.gc_modules):
            gc.t = np.concatenate(gc.get_s())

    def set_as_target_state(self, gc_connections):
        for m, gc in enumerate(self.gc_modules):
            gc.t = gc_connections[m]
        print("Set new target state")

    def reset_s_virtual(self):
        for m, gc in enumerate(self.gc_modules):
            gc.reset_s_virtual()

    def set_filename_as_target_state(self, filename):
        t_vectors = np.load("data/gc_model/" + filename)
        for m, gc in enumerate(self.gc_modules):
            gc.t = t_vectors[m]
        print("Set loaded data as new target state:", filename)

    def plot_modules(self, nr_steps):
        plot_3D_sheets(self.gc_modules, nr_steps)
