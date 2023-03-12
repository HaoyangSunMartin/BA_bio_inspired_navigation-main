
import numpy as np


def check_tuned_direction_vector(gc_network):
    direction_vector = []
    new_dim = gc_network.n
    for gc in gc_network.gc_modules:
        s = gc.get_s(virtual=False)
        filter = np.where(s>0.1, 1.0, 0.0)

        s = np.multiply(s, filter)
        print(s.shape)

        #s = np.reshape(s,(new_dim,new_dim))

        #s_filtered = np.where(s>0.1, 1, 0)
        decided = False
        for axis in range(2):
            dir = 'x' if axis==0 else 'y'
            pro = np.sum(s, axis=axis)
            #print("current direction is ", dir, " argmin is ", np.amin(pro))
            if np.amin(pro) == 0 and not decided:
                direction_vector.append(dir)
                decided = True
        if not decided:
            direction_vector.append('u')

    return direction_vector

def naive_average_smoothing(array):
    c = np.copy(array)
    for i in range(len(array)):
        if i==0 or i==len(array)-1:
            continue
        else:
            c[i]=np.average([array[i-1],array[i], array[i+1]])
    return c
def naive_2D_sheet_smoothing(sheet,n):
    sheet_copy = np.copy(sheet)
    for x, _ in enumerate(sheet):
        for y, _ in enumerate(sheet[x]):
            l = []
            l.append(sheet[x][y])
            if x != 0:
                l.append(sheet[x - 1][y])

            if x != n - 1:
                l.append(sheet[x + 1][y])

            if y != 0:
                l.append(sheet[x][y - 1])

            if y != n - 1:
                l.append(sheet[x][y + 1])
            sheet_copy[x][y]=np.average(l)
    return sheet_copy
#this function computes the relative angle between 2 vectors
def compute_angle(vec_1, vec_2):
    length_vector_1 = np.linalg.norm(vec_1)
    length_vector_2 = np.linalg.norm(vec_2)
    if length_vector_1 == 0 or length_vector_2 == 0:
        return 0
    unit_vector_1 = vec_1 / length_vector_1
    unit_vector_2 = vec_2 / length_vector_2
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    vec = np.cross([vec_1[0], vec_1[1], 0], [vec_2[0], vec_2[1], 0])

    return angle * np.sign(vec[2])


def compute_theta(vec):
    if vec[0] == 0:
        angle = np.pi/2
    else:
        angle = np.arctan(abs(vec[1] / vec[0]))
        if vec[0] < 0:
            angle = np.pi - angle
    return angle * np.sign(vec[1])


def compute_axis_limits(arena_size, xy_coordinates=None, environment=None):
    temp_arena_size = 1.1 * arena_size
    limits_t = [- temp_arena_size, temp_arena_size,
                - temp_arena_size, temp_arena_size]
    if environment == "linear_sunburst":
        limits_t = [0, arena_size,
                    0, arena_size]
    if xy_coordinates is not None:
        # Compute Axis limits for plot
        x, y = zip(*xy_coordinates)
        limits_t = [np.around(min(x), 1) - 0.1, np.around(max(x), 1) + 0.1,
                    np.around(min(y), 1) - 0.1, np.around(max(y), 1) + 0.1]

    x_t_width = limits_t[1] - limits_t[0]
    y_t_width = limits_t[3] - limits_t[2]

    x_width = 432.0
    y_width = 306.0
    ratio = x_width / y_width
    if ratio >= x_t_width / y_t_width:
        rescaled_width = y_t_width * ratio
        diff = (rescaled_width - x_t_width) / 2
        limits_t[0] = limits_t[0] - diff
        limits_t[1] = limits_t[1] + diff
    else:
        rescaled_width = x_t_width / ratio
        diff = (rescaled_width - y_t_width) / 2
        limits_t[2] = limits_t[2] - diff
        limits_t[3] = limits_t[3] + diff

    return limits_t

### changes by Haoyang Sun- Start
global timer4LinearLookAhead
timer4LinearLookAhead = 0.0
### changes by Haoyang Sun- End


