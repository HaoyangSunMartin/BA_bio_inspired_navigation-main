
from system.helper import compute_angle
from system.decoder.spikeDetection import SpikeDetector

import matplotlib.colors as mcolors
import operator
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
mpl.rcParams['animation.ffmpeg_path'] = "ffmpeg/ffmpeg"


TUM_colors = {
                'TUMBlue': '#0065BD',
                'TUMSecondaryBlue': '#005293',
                'TUMSecondaryBlue2': '#003359',
                'TUMBlack': '#000000',
                'TUMWhite': '#FFFFFF',
                'TUMDarkGray': '#333333',
                'TUMGray': '#808080',
                'TUMLightGray': '#CCCCC6',
                'TUMAccentGray': '#DAD7CB',
                'TUMAccentOrange': '#E37222',
                'TUMAccentGreen': '#A2AD00',
                'TUMAccentLightBlue': '#98C6EA',
                'TUMAccentBlue': '#64A0C8'
}

cmap_binary = mcolors.ListedColormap([TUM_colors['TUMWhite'], TUM_colors['TUMGray']])

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(256/256, 0/256, N)
vals[:, 1] = np.linspace(256/256, 101/256, N)
vals[:, 2] = np.linspace(256/256, 189/256, N)
tum_blue_map = mcolors.ListedColormap(vals)

vals2 = np.ones((N, 4))
vals2[:, 0] = np.linspace(256/256, 128/256, N)
vals2[:, 1] = np.linspace(256/256, 128/256, N)
vals2[:, 2] = np.linspace(256/256, 128/256, N)
tum_grey_map = mcolors.ListedColormap(vals2)

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


# --------------- Plot grid cells ---------------
def plot_grid_cell_modules(gc_modules, i, plot_target=False, plot_matches=False):

    spike_detector = None
    if plot_matches:
        spike_detector = SpikeDetector()
        spike_detector.compute_direction_signal(gc_modules, 0)

    fig = plt.figure(figsize=(12, 4))

    for m, gc in enumerate(gc_modules):
        s = np.reshape(gc.s, (gc.n, gc.n))
        t = np.reshape(gc.t, (gc.n, gc.n))

        ax = fig.add_subplot(1, len(gc_modules), m + 1)
        ax.axes.get_xaxis().set_visible(False)

        if m != 0:
            ax.axes.get_yaxis().set_visible(False)

        # title_string = "g_m = " + str("{:.2f}".format(gc.gm))
        plt.title(r"$g_m =$" + " " + "{:.2f}".format(gc.gm))

        plt.imshow(s, origin="lower", cmap=tum_blue_map)

        if plot_target:
            plt.imshow(t, alpha=0.5, cmap=tum_grey_map, origin="lower")

        if plot_matches:
            matches = spike_detector.matches_dict[0][m]
            vectors = spike_detector.vector_dict[0][m]

            if len(matches) != 0 and len(vectors) != 0:
                s_max = list(matches.keys())
                s_max_x, s_max_y = zip(*s_max)
                t_max = list(matches.values())
                t_max_x, t_max_y = zip(*t_max)

                origin_x, origin_y = zip(*list(vectors.keys()))
                vectors_x, vectors_y = zip(*list(vectors.values()))

                plt.scatter(s_max_x, s_max_y, color=TUM_colors['TUMBlue'], s=1)
                plt.scatter(t_max_x, t_max_y, color=TUM_colors['TUMGray'], s=1)

                plt.quiver(origin_x, origin_y, vectors_x, vectors_y, color=TUM_colors['TUMDarkGray'],
                           width=0.01, scale=1, scale_units='xy')

    # folder = "spikes_matched/" if plot_matches else "spikes_unmatched/"
    folder = ""

    directory = "experiments/grid_cell_initialization/" + folder
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig("experiments/grid_cell_initialization/" + folder + "grid_cell_initialization_" + str(i), format="pdf")

    plt.show(block=False)
    plt.close()


def plot_3D_sheets(gc_modules, i ):

    fig = plt.figure(figsize=(12, 4))

    for idx, gc in enumerate(gc_modules):
        #this if statement decides the shape of the s vector(also if cuda is used)
        if len(gc.get_s(virtual=False).shape) == 2:
            sheet = gc.get_s(virtual=False)
        else:
            n = int(np.sqrt(len(gc.get_s(False))))
            sheet = np.reshape(gc.get_s(False), (n, n))

        xmin, xmax, nx = 0, sheet.shape[0] - 1, sheet.shape[0]
        ymin, ymax, ny = 0, sheet.shape[1] - 1, sheet.shape[1]
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)

        ax = fig.add_subplot(1, int(len(gc_modules)), idx + 1, projection='3d')
        ax.plot_surface(X, Y, sheet, cmap=tum_blue_map, shade=True)
        ax.set_zlim(0, 1)

        major_ticks = np.linspace(0, 1, 3, endpoint=True)
        ax.set_zticks(major_ticks)

        if idx != len(gc_modules) - 1:
            ax.set_zticklabels([])

        plt.title(r"$g_m =$" + " " + "{:.2f}".format(gc.gm))

    directory = "experiments/grid_cell_initialization/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig("experiments/grid_cell_initialization/" + "grid_cell_initialization_3D_" + str(i), format="pdf")
    plt.show(block=False)
    plt.close()


# --------------- Plot linear lookahead functionality ---------------
def plot_vectors(s, t, axis=0, i=0, found=False):

    fig = plt.figure()
    n = int(np.sqrt(len(s)))
    s = np.reshape(s, (n, n))
    t = np.reshape(t, (n, n))

    plt.imshow(s, origin="lower", cmap=tum_blue_map)
    plt.imshow(t, origin="lower", cmap=cmap_binary, alpha=0.5)
    # plt.show()

    directory = "plots/linear_lookahead/found/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if found:
        plt.savefig("plots/linear_lookahead/found/grid_cells_linear_lookahead_" + str(axis) + str(i))
    else:
        plt.savefig("plots/linear_lookahead/grid_cells_linear_lookahead_" + str(axis) + str(i))
    plt.close()


def plot_linear_lookahead_function(proj_gc_connections, proj_s_vectors, filtered, axis=0, found=False):
    for idx, p in enumerate(proj_gc_connections):
        fig = plt.figure()
        x = np.linspace(0, 40, num=40)
        fig.add_subplot(4, 1, 1)
        plt.bar(x, proj_gc_connections[idx], 1.1, color=TUM_colors['TUMGray'])
        plt.bar(x, proj_s_vectors[idx], 1.1, color=TUM_colors['TUMBlue'], alpha=0.5)
        fig.add_subplot(4, 1, 2)
        plt.bar(x, proj_gc_connections[idx], 1.1, color=TUM_colors['TUMGray'])
        fig.add_subplot(4, 1, 3)
        plt.bar(x, proj_s_vectors[idx], 1.1, color=TUM_colors['TUMBlue'])
        fig.add_subplot(4, 1, 4)
        plt.bar(x, filtered[idx], 1.1,  color=TUM_colors['TUMSecondaryBlue2'])
        # plt.show()

        directory = "plots/linear_lookahead/found/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        if found:
            plt.savefig("plots/linear_lookahead/found/projection_linear_lookahead_" + str(axis) + str(idx))
        else:
            plt.savefig("plots/linear_lookahead/projection_linear_lookahead_" + str(axis) + str(idx))
        plt.close()

# --------------- Plot Linear Look Ahead Axis Projection ---------------
def plot_LLA_Projection(distance, firing, axis):
    fig, ax = plt.subplots()
    ax.set_xlabel("projective distance")
    ax.set_ylabel("projective firing")
    direction = 'x' if axis==0 else 'y'
    ax.set_title("Projective LLA on "+ direction+ " Axis")
    ax.plot(distance, firing, 'o')



# --------------- Plot grid cell decoder all trials ---------------
def plot_vector_navigation_error(error):

    range_max = 2.5
    plt.hist(error, range=(0, range_max), rwidth=0.9, bins=50,
             weights=np.ones_like(error) / error.size, color=TUM_colors['TUMBlue'])
    plt.axvline(error.mean(), color=TUM_colors['TUMGray'], linestyle='dashed', linewidth=2)
    plt.annotate("mean error\n" + "{:.3f}".format(error.mean()), (error.mean() + 0.11 * range_max, 0.12), ha='left')
    plt.title("Cognitive-map and linear look-ahead precision")
    plt.xlabel("Absolute error between actual position and estimated position [m]")
    plt.ylabel("Relative frequency")
    plt.savefig("experiments/" + "errors", format="pdf")
    plt.show(block=False)
    plt.close()


def plot_vector_angle_error(positions, vectors):

    error = np.zeros((len(positions), 1))
    for idx, pos in enumerate(positions):
        angle = compute_angle(-positions[idx], vectors[idx])
        error[idx] = abs(angle) * 360 / (2 * np.pi)

    range_max = 30
    plt.hist(error, range=(0, range_max), rwidth=0.9, bins=50,
             weights=np.ones_like(error) / error.size, color=TUM_colors['TUMBlue'])
    plt.axvline(error.mean(), color=TUM_colors['TUMGray'], linestyle='dashed', linewidth=2)
    plt.annotate("mean error\n" + "{:.3f}".format(error.mean()), (error.mean() + 0.02 * range_max, 0.17), ha='left')
    plt.title("Spike-detection angle precision")
    plt.xlabel("Absolute error in decoded vector angle [°]")
    plt.ylabel("Relative frequency")
    plt.savefig("experiments/" + "errors_angle", format="pdf")
    plt.show(block=False)
    plt.close()


def plot_vector_distance_error(positions, vectors):

    true_distance = np.linalg.norm(positions, axis=1)
    computed_distance = np.linalg.norm(vectors, axis=1)

    error = abs(true_distance - computed_distance)

    range_max = 2.5
    plt.hist(error, range=(0, range_max), rwidth=0.9, bins=50,
             weights=np.ones_like(error) / error.size, color=TUM_colors['TUMBlue'])
    plt.axvline(error.mean(), color=TUM_colors['TUMGray'], linestyle='dashed', linewidth=2)
    plt.annotate("mean error\n" + "{:.3f}".format(error.mean()), (error.mean() + 0.08 * range_max, 0.11), ha='left')
    plt.title("Spike-detection distance precision")
    plt.xlabel("Absolute error in decoded vector distance [m]")
    plt.ylabel("Relative frequency")
    plt.savefig("experiments/" + "errors_distance", format="pdf")
    plt.show(block=False)
    plt.close()


def plot_positions(positions, vectors):

    goal_positions = positions + vectors
    x0, y0 = zip(*positions)
    x1, y1 = zip(*goal_positions)
    dx, dy = zip(*vectors)

    plt.figure()
    circle = plt.Circle((0, 0), 0.5, color=TUM_colors['TUMGray'], alpha=0.3)
    plt.gca().add_patch(circle)
    plt.quiver(x0, y0, dx, dy, angles='xy', scale_units='xy', scale=1,
               color=TUM_colors['TUMGray'], alpha=0.4, width=0.005)
    plt.scatter(x1, y1, color=TUM_colors['TUMBlue'], alpha=0.4)
    plt.scatter(x0, y0, color=TUM_colors['TUMDarkGray'], s=100)
    plt.axis('square')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.savefig("experiments/" + "vectors", format="pdf")
    plt.show(block=False)
    plt.close()


# --------------- Plot grid cell decoder single trial ---------------
def plot_error_single_run(positions, vectors):

    errors = np.linalg.norm(positions[3501:] + vectors[3501:], axis=1)

    plt.plot(errors)
    plt.show(block=False)


def plot_angles_single_run(positions, vectors):

    x0, y0 = zip(*(positions * -1))
    x1, y1 = zip(*vectors)

    true_angle = np.arctan2(x0[3501:], y0[3501:])
    computed_angle = np.arctan2(x1[3501:], y1[3501:])

    plt.plot(true_angle)
    plt.plot(computed_angle)

    plt.ylim(-np.pi, np.pi)

    plt.legend(["true angle", "computed angle"])
    plt.show(block=False)


def plot_distances_single_run(positions, vectors):

    true_distance = np.linalg.norm(positions[3501:], axis=1)
    computed_distance = np.linalg.norm(vectors[3501:], axis=1)

    plt.plot(true_distance, color=TUM_colors['TUMGray'])
    plt.plot(computed_distance, color=TUM_colors['TUMBlue'])

    plt.legend(["true distance", "computed distance"])
    plt.title("Sample run of agent heading back to goal")
    plt.xlabel("time-steps in navigation phase")
    plt.ylabel(r"goal vector length $[m]$")
    plt.savefig("experiments/" + "spike_detector_mse_sample_run", format="pdf")
    plt.show(block=False)


def plot_mean_squared_error(mse_dict):
    plt.plot(mse_dict.keys(), mse_dict.values(), color=TUM_colors['TUMBlue'])

    factor_best = min(mse_dict.items(), key=operator.itemgetter(1))[0]
    plt.axvline(factor_best, color=TUM_colors['TUMGray'], linestyle='dashed', linewidth=2)
    plt.annotate("optimal factor\n" + "{:.3f}".format(factor_best), (factor_best + 0.02 * 1, 75), ha='left')

    plt.title("Spike-detector parameter search")
    plt.xlabel(r"scaling vector $\rho$")
    plt.ylabel(r"mean squared error of all $\delta$ $[m^2]$")

    plt.savefig("experiments/" + "spike_detector_mse", format="pdf")
    plt.show(block=False)


# --------------- Plot cognitive map ---------------
def cognitive_map_plot(pc_network, cognitive_map, vectors_array=None, env_coding="plane", center_block = False):

    plt.figure()

    ax = plt.gca()
    add_cognitive_map(ax, pc_network, cognitive_map)

    if vectors_array is not None:
        for vec in vectors_array:
            plt.quiver(5.5, 0.5, vec[0], vec[1], color=TUM_colors['TUMLightGray'],
                       angles='xy', scale_units='xy', scale=1)

    # Plot obstacles
    add_environment(ax,env_coding, center_block= center_block )
    plt.axis('square')
    plt.xlim(-0.5, 11.5)
    plt.ylim(-0.5, 11.5)
    plt.savefig("experiments/" + "cognitive_map_exploration_phase", format="pdf")
    plt.show()


def add_cognitive_map(ax, pc_network, cognitive_map):
    for i, pc in enumerate(pc_network.place_cells):

        for j, connection in enumerate(cognitive_map.topology_cells[i]):
            if connection == 1 and i != j:
                x_values = [pc.env_coordinates[0], pc_network.place_cells[j].env_coordinates[0]]
                y_values = [pc.env_coordinates[1], pc_network.place_cells[j].env_coordinates[1]]
                ax.plot(x_values, y_values, color=TUM_colors['TUMGray'], alpha=0.2)
        ###changes by Haoyang Sun--Start
        #annotate the place cells with their IDs
        x_coord, y_coord = pc.env_coordinates
        ax.text(x_coord,y_coord, str(i), fontsize=10)

        ###changes by Haoyang Sun--End

        circle = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.5,
                            fc=TUM_colors['TUMBlue'], alpha=cognitive_map.reward_cells[i] * 0.6,
                            ec=TUM_colors['TUMGray'], linewidth=0)
        ax.add_artist(circle)
        circle_border = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.5,
                                ec=TUM_colors['TUMGray'], fill=False, linewidth=0.3)
        ax.add_artist(circle_border)

        circle = plt.Circle((1.5, 10), 0.12,
                            fc=TUM_colors['TUMBlue'],
                            ec=TUM_colors['TUMGray'], linewidth=0)
        ax.add_artist(circle)



def add_block_cells(ax, bc_list):
    for i, bc in enumerate(bc_list.block_cells):


        ###changes by Haoyang Sun--Start
        #annotate the place cells with their IDs
        x_coord, y_coord = bc.env_coordinates
        ax.text(x_coord,y_coord, str(i), fontsize=10)

        ###changes by Haoyang Sun--End
        ax.plot(x_coord, y_coord, 'r+')

def add_cells_firing_map(ax,pc_network, cognitive_map, firing_vector ):
    for i, pc in enumerate(pc_network.place_cells):

        for j, connection in enumerate(cognitive_map.topology_cells[i]):
            if connection == 1 and i != j:
                x_values = [pc.env_coordinates[0], pc_network.place_cells[j].env_coordinates[0]]
                y_values = [pc.env_coordinates[1], pc_network.place_cells[j].env_coordinates[1]]
                ax.plot(x_values, y_values, color=TUM_colors['TUMGray'], alpha=0.2)
        ###changes by Haoyang Sun--Start
        #annotate the place cells with their firing values
        x_coord, y_coord = pc.env_coordinates
        ax.text(x_coord,y_coord, str(firing_vector[i]), fontsize=10)
        ###changes by Haoyang Sun--End

        circle = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.5,
                            fc=TUM_colors['TUMBlue'], alpha=cognitive_map.reward_cells[i] * 0.6,
                            ec=TUM_colors['TUMGray'], linewidth=0)
        ax.add_artist(circle)
        circle_border = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.5,
                                ec=TUM_colors['TUMGray'], fill=False, linewidth=0.3)
        ax.add_artist(circle_border)

        circle = plt.Circle((1.5, 10), 0.12,
                            fc=TUM_colors['TUMBlue'],
                            ec=TUM_colors['TUMGray'], linewidth=0)
        ax.add_artist(circle)





def add_environment(ax, env="plane", center_block=False):
    dic = {
        "plane_doors"  : [1, 3, 5, 7],
        "plane_doors_1" : [3, 5, 7],
        "plane_doors_2" : [1, 5, 7],
        "plane_doors_3" : [1, 3, 7],
        "plane_doors_4" : [1, 3, 5],
        "plane_doors_5c_3o" : [1, 3, 7, 9],
        "plane": [],
        "plane_unstructured": [],
        "plane_doors_huge": [],
        "plane_doors_Scale_2": []
    }
    if center_block:
        plot_box = plt.Rectangle((2,2.5 ), 8, 1, color=TUM_colors['TUMLightGray'])
        ax.add_artist(plot_box)
    doors = dic[env]#[1, 3, 5, 7]# [1, 3, 5, 7]
    triangel_y=None
    box_y =None
    door_y = None
    if env=="plane_unstructured":
        #the unstrctured map is: for triangels: 5 6 5 4 5 6
        #                        for boxes: 8.5 9.0 8.0 8.0 9.0
        triangel_y = [4.0,5.0,4.0,3.0,4.0,5.0]
        box_y =[8,8.5,7.5,7.5,8.5]
        door_y = []
    else:
        triangel_y = [5,5,5,5,5,5]
        box_y = [8,8,8,8,8]
        door_y = []

    for x in doors:
        plot_box = plt.Rectangle((x, 5.4), 1, 0.2, color=TUM_colors['TUMGray'])
        ax.add_artist(plot_box)
    boxes = [0, 2, 4, 6, 8, 10]
    for i, x in enumerate(boxes):
        plot_box = plt.Rectangle((x, triangel_y[i]), 1, 2, color=TUM_colors['TUMLightGray'])
        ax.add_artist(plot_box)
    boxes = [1, 3, 5, 7, 9]
    for i, x in enumerate(boxes):
        plot_box = plt.Rectangle((x, box_y[i]), 1, 1, color=TUM_colors['TUMLightGray'])
        ax.add_artist(plot_box)
    plot_box = plt.Rectangle((-0.1, -0.1), 11.2, 11.2, color=TUM_colors['TUMLightGray'], fc='none',
                             ec=TUM_colors['TUMLightGray'], linewidth=5)
    ax.add_artist(plot_box)

def add_goal(ax):
    circle = plt.Circle((1.5, 10), 0.12,
                        fc=TUM_colors['TUMBlue'],
                        ec=TUM_colors['TUMGray'], linewidth=0)
    ax.add_artist(circle)



def add_trajectory(ax, xy_coordinates):
    x_values = []
    y_values = []
    for x, y in xy_coordinates:
        x_values.insert(0, x)
        y_values.insert(0, y)

    ax.plot(x_values, y_values, color=TUM_colors['TUMBlue'], alpha=1.0)



# --------------- Plot sub goal localization map ---------------
def plot_sub_goal_localization(env, cognitive_map, pc_network, goal_vector, filename, env_coding="plane", chosen_idx=0, goal_spiking=None, center_block = False):

    plt.figure()

    xy_coordinates = env.xy_coordinates

    ax = plt.gca()
    add_cognitive_map(ax, pc_network, cognitive_map)
    add_environment(ax,env_coding, center_block=center_block)

    initial = plt.Circle((5.5, 0.5), 0.12, color=TUM_colors['TUMGray'])
    ax.add_artist(initial)

    current_position = xy_coordinates[-1]

    if goal_spiking is not None:
        for idx, angle in enumerate(goal_spiking):

            if idx % 4 == 0:
                color = TUM_colors['TUMDarkGray']
                length = 0.75

                if idx == chosen_idx:
                    color = TUM_colors['TUMBlue']
                    length = np.maximum(goal_spiking[angle]["distance"], 0.5)
                if goal_spiking[angle]["blocked"]:
                    color = TUM_colors['TUMGray']

                vector = np.array([np.cos(angle), np.sin(angle)]) * length
                ax.quiver(current_position[0], current_position[1], vector[0], vector[1],
                           color=color, angles='xy', scale_units='xy', scale=1)

    agent = plt.Circle((current_position[0], current_position[1]), 0.25, color=TUM_colors['TUMDarkGray'])
    ax.add_artist(agent)
    angle = env.orientation_angle[-1]
    ax.quiver(current_position[0], current_position[1], np.cos(angle) * 0.4, np.sin(angle) * 0.4,
              color=TUM_colors['TUMDarkGray'], headwidth=7,
              angles='xy', scale_units='xy', scale=1)

    if not env.topology_based or (abs(goal_vector[0]) > 0.5 and abs(goal_vector[1]) > 0.5):
        ax.quiver(current_position[0], current_position[1], goal_vector[0], goal_vector[1],
                  color=TUM_colors['TUMDarkGray'], angles='xy', scale_units='xy', scale=1,
                  width=0.01)

    plt.axis('square')
    plt.xlim(-0.5, 11.5)
    plt.ylim(-0.5, 11.5)
    plt.savefig("experiments/" + "goal_lookahead" + filename, format="pdf")
    plt.show(block=False)
    plt.close()
def calculate_trajectory_distance(xy_coordinates):
    distance = 0.0
    prev = xy_coordinates[0]
    for position in xy_coordinates[1::]:
        dist = [position[0]-prev[0],position[1]-prev[1]]
        distance = distance + np.linalg.norm(dist)
        prev = position
    return distance



def plot_trajectory(xy_coordinates, door,env_coding="plane", center_block=False):

    # print("----- Trajectory " + door)
    #
    # distance = 0
    # for idx in range(len(xy_coordinates) - 1):
    #     vec = xy_coordinates[idx + 1] - xy_coordinates[idx]
    #     distance = distance + np.linalg.norm(vec)
    # print("Overall distance: ", distance)

    plt.figure()

    ax = plt.gca()
    add_environment(ax, env_coding, center_block=center_block)

    initial = plt.Circle((5.5, 0.5), 0.12, color=TUM_colors['TUMGray'])
    ax.add_artist(initial)

    # xy_coordinates = xy_coordinates[:5350]
    # x, y = zip(*xy_coordinates)
    # plt.scatter(x, y, s=1.5, c=TUM_colors['TUMBlue'])
    #
    # distance = 0
    # for idx in range(len(xy_coordinates) - 1):
    #     vec = xy_coordinates[idx + 1] - xy_coordinates[idx]
    #     distance = distance + np.linalg.norm(vec)
    # print("Until <0.5: ", distance)
    #
    # # points = np.array([[5.5, 0.5], [9.5, 5], [9.5, 7.5], [2.5, 7.5], [2.5, 9], [1.5, 10]])  # 5 open
    # # points = np.array([[5.5, 0.5], [3.5, 5], [3.5, 7.5], [2.5, 7.5], [2.5, 9], [1.5, 10]])  # all open
    # # points = np.array([[5.5, 0.5], [5.5, 7.5], [2.5, 7.5], [2.5, 9], [1.5, 10]])  # 3 open
    # points = np.array([[5.5, 0.5], [1.5, 5], [1.5, 7.5], [2.5, 7.5], [2.5, 9], [1.5, 10]])  # 1 open
    #
    # distance = 0
    # for idx in range(len(points) - 1):
    #     vec = points[idx + 1] - points[idx]
    #     distance = distance + np.linalg.norm(vec)
    #     if idx == 0:
    #         points[idx] = points[idx] + (vec/np.linalg.norm(vec)) * 0.25
    #     x, y = zip(*[points[idx], points[idx + 1]])
    #     plt.plot(x, y, ':', color=TUM_colors['TUMLightGray'])
    # print("Optimal distance: ", distance)

    current_position = xy_coordinates[0]
    agent = plt.Circle((current_position[0], current_position[1]), 0.25, color=TUM_colors['TUMDarkGray'])
    ax.add_artist(agent)
    angle = np.pi / 2
    ax.quiver(current_position[0], current_position[1], np.cos(angle) * 0.4, np.sin(angle) * 0.4,
              color=TUM_colors['TUMDarkGray'], headwidth=7,
              angles='xy', scale_units='xy', scale=1)

    circle = plt.Circle((1.5, 10), 0.12,
                        fc=TUM_colors['TUMBlue'],
                        ec=TUM_colors['TUMGray'], linewidth=0)
    ax.add_artist(circle)

    circle = plt.Circle((1.5, 10), 0.5,
                        fc=TUM_colors['TUMBlue'], alpha=0.6,
                        ec=TUM_colors['TUMGray'], linewidth=0)
    ax.add_artist(circle)

    plt.axis('square')
    plt.xlim(-0.5, 11.5)
    plt.ylim(-0.5, 11.5)
    plt.savefig("experiments/maze_navigation/" + door + "/plots/" + "trajectory_initial", format="pdf")
    plt.show()
    plt.close()


def plot_cognitive_map(env, cognitive_map, pc_network, goal_vector, filename,env_coding="plane", chosen_idx=0, goal_spiking=None, center_block=False):
    plt.figure()

    xy_coordinates = env.xy_coordinates

    ax = plt.gca()
    add_cognitive_map(ax, pc_network, cognitive_map)
    add_environment(ax,env_coding,center_block=center_block)
    add_goal(ax)
    initial = plt.Circle((5.5, 0.5), 0.12, color=TUM_colors['TUMGray'])
    ax.add_artist(initial)

    current_position = [5.5, 0.5]##xy_coordinates[-1]

    if goal_spiking is not None:
        for idx, angle in enumerate(goal_spiking):

            if idx % 4 == 0:
                color = TUM_colors['TUMDarkGray']
                length = 0.75

                if idx == chosen_idx:
                    color = TUM_colors['TUMBlue']
                    length = np.maximum(goal_spiking[angle]["distance"], 0.5)
                if goal_spiking[angle]["blocked"]:
                    color = TUM_colors['TUMGray']

                vector = np.array([np.cos(angle), np.sin(angle)]) * length
                ax.quiver(current_position[0], current_position[1], vector[0], vector[1],
                          color=color, angles='xy', scale_units='xy', scale=1)

    agent = plt.Circle((current_position[0], current_position[1]), 0.25, color=TUM_colors['TUMDarkGray'])
    ax.add_artist(agent)
    angle = env.orientation_angle[-1]
    ax.quiver(current_position[0], current_position[1], np.cos(angle) * 0.4, np.sin(angle) * 0.4,
              color=TUM_colors['TUMDarkGray'], headwidth=7,
              angles='xy', scale_units='xy', scale=1)
    """
    if not env.topology_based or (abs(goal_vector[0]) > 0.5 and abs(goal_vector[1]) > 0.5):
        ax.quiver(current_position[0], current_position[1], goal_vector[0], goal_vector[1],
                  color=TUM_colors['TUMDarkGray'], angles='xy', scale_units='xy', scale=1,
                  width=0.01)
    """
    plt.axis('square')
    plt.xlim(-0.5, 11.5)
    plt.ylim(-0.5, 11.5)
    plt.savefig("experiments/" + "goal_lookahead" + filename, format="pdf")
    plt.show()
    plt.close()

def plot_trajectory_on_map(xy_coordinates,env, cognitive_map, pc_network, goal_vector, filename,env_coding="plane", chosen_idx=0, goal_spiking=None, center_block=False):
    plt.figure()

    #xy_coordinates = env.xy_coordinates

    ax = plt.gca()
    add_cognitive_map(ax, pc_network, cognitive_map)
    add_environment(ax, env_coding, center_block=center_block)
    add_trajectory(ax,xy_coordinates)


    initial = plt.Circle((5.5, 0.5), 0.12, color=TUM_colors['TUMGray'])
    ax.add_artist(initial)

    current_position = [5.5, 0.5]  ##xy_coordinates[-1]

    if goal_spiking is not None:
        for idx, angle in enumerate(goal_spiking):

            if idx % 4 == 0:
                color = TUM_colors['TUMDarkGray']
                length = 0.75

                if idx == chosen_idx:
                    color = TUM_colors['TUMBlue']
                    length = np.maximum(goal_spiking[angle]["distance"], 0.5)
                if goal_spiking[angle]["blocked"]:
                    color = TUM_colors['TUMGray']

                vector = np.array([np.cos(angle), np.sin(angle)]) * length
                ax.quiver(current_position[0], current_position[1], vector[0], vector[1],
                          color=color, angles='xy', scale_units='xy', scale=1)

    agent = plt.Circle((current_position[0], current_position[1]), 0.25, color=TUM_colors['TUMDarkGray'])
    ax.add_artist(agent)
    angle = env.orientation_angle[-1]
    ax.quiver(current_position[0], current_position[1], np.cos(angle) * 0.4, np.sin(angle) * 0.4,
              color=TUM_colors['TUMDarkGray'], headwidth=7,
              angles='xy', scale_units='xy', scale=1)
    """
    if not env.topology_based or (abs(goal_vector[0]) > 0.5 and abs(goal_vector[1]) > 0.5):
        ax.quiver(current_position[0], current_position[1], goal_vector[0], goal_vector[1],
                  color=TUM_colors['TUMDarkGray'], angles='xy', scale_units='xy', scale=1,
                  width=0.01)
    """
    plt.axis('square')
    plt.xlim(-0.5, 11.5)
    plt.ylim(-0.5, 11.5)
    #plt.savefig("experiments/" + "goal_lookahead" + filename, format="pdf")
    plt.show()
    plt.close()

def plot_block_cells_on_map(env, cognitive_map, bc_list, goal_vector, filename,center_block=False,env_coding="plane", chosen_idx=0, goal_spiking=None):
    plt.figure()

    # xy_coordinates = env.xy_coordinates

    ax = plt.gca()
    add_block_cells(ax, bc_list)
    add_environment(ax, env_coding, center_block=center_block)


def plot_cognitive_map_with_bc(env, cognitive_map, bc_list, pc_network, goal_vector, filename,center_block=False,env_coding="plane", chosen_idx=0, goal_spiking=None):
    plt.figure()

    xy_coordinates = env.xy_coordinates

    ax = plt.gca()
    add_cognitive_map(ax, pc_network, cognitive_map)
    add_environment(ax, env_coding,center_block=center_block)
    add_block_cells(ax, bc_list)

    initial = plt.Circle((5.5, 0.5), 0.12, color=TUM_colors['TUMGray'])
    ax.add_artist(initial)

    current_position = [5.5, 0.5]##xy_coordinates[-1]

    if goal_spiking is not None:
        for idx, angle in enumerate(goal_spiking):

            if idx % 4 == 0:
                color = TUM_colors['TUMDarkGray']
                length = 0.75

                if idx == chosen_idx:
                    color = TUM_colors['TUMBlue']
                    length = np.maximum(goal_spiking[angle]["distance"], 0.5)
                if goal_spiking[angle]["blocked"]:
                    color = TUM_colors['TUMGray']

                vector = np.array([np.cos(angle), np.sin(angle)]) * length
                ax.quiver(current_position[0], current_position[1], vector[0], vector[1],
                          color=color, angles='xy', scale_units='xy', scale=1)

    agent = plt.Circle((current_position[0], current_position[1]), 0.25, color=TUM_colors['TUMDarkGray'])
    ax.add_artist(agent)
    angle = env.orientation_angle[-1]
    ax.quiver(current_position[0], current_position[1], np.cos(angle) * 0.4, np.sin(angle) * 0.4,
              color=TUM_colors['TUMDarkGray'], headwidth=7,
              angles='xy', scale_units='xy', scale=1)
    """
    if not env.topology_based or (abs(goal_vector[0]) > 0.5 and abs(goal_vector[1]) > 0.5):
        ax.quiver(current_position[0], current_position[1], goal_vector[0], goal_vector[1],
                  color=TUM_colors['TUMDarkGray'], angles='xy', scale_units='xy', scale=1,
                  width=0.01)
    """
    plt.axis('square')
    plt.xlim(-0.5, 11.5)
    plt.ylim(-0.5, 11.5)
    plt.savefig("experiments/" + "goal_lookahead" + filename, format="pdf")
    plt.show(block=False)
    plt.close()
def plot_env_map_with_bc(bc_list,env, center_block=False,env_coding="plane"):
    plt.figure()

    ax = plt.gca()
    add_environment(ax, env_coding, center_block= center_block)
    add_block_cells(ax, bc_list)

    current_position = env.xy_coordinates[-1]

    agent = plt.Circle((current_position[0], current_position[1]), 0.25, color=TUM_colors['TUMDarkGray'])

    ax.add_artist(agent)
    angle = env.orientation_angle[-1]
    ax.quiver(current_position[0], current_position[1], np.cos(angle) * 0.4, np.sin(angle) * 0.4,
              color=TUM_colors['TUMDarkGray'], headwidth=7,
              angles='xy', scale_units='xy', scale=1)
    """
    if not env.topology_based or (abs(goal_vector[0]) > 0.5 and abs(goal_vector[1]) > 0.5):
        ax.quiver(current_position[0], current_position[1], goal_vector[0], goal_vector[1],
                  color=TUM_colors['TUMDarkGray'], angles='xy', scale_units='xy', scale=1,
                  width=0.01)
    """
    plt.axis('square')
    plt.xlim(-0.5, 11.5)
    plt.ylim(-0.5, 11.5)

    plt.show(block=False)
    plt.close()

def plot_step_time(record):
    names = range(len(record))
    values = record
    fig, ax = plt.subplots()
    ax.set_xlabel("step")
    ax.set_ylabel("time consumption")
    plt.bar(names, values)
    plt.suptitle("Time Consumption of each step")
    plt.yscale("log")
    plt.show()

def plot_Time_Comparison(names, record, title=None):

    values = record
    fig, ax = plt.subplots()
    ax.set_xlabel("Methods")
    ax.set_ylabel("Time Consumption (seconds)")
    plt.bar(names, values)
    if title is None:
        plt.suptitle("Time Consumption Comparison")
    else:
        plt.suptitle(title)
    plt.yscale("log")
    plt.show()
def plot_Path_Length_Comparison(names, record, title=None):

    values = record
    fig, ax = plt.subplots()
    ax.set_xlabel("Methods")
    ax.set_ylabel("Time Consumption (meters)")
    plt.bar(names, values)
    if title is None:
        plt.suptitle("Path Length Comparison")
    else:
        plt.suptitle(title)
    plt.yscale("log")
    plt.show()

def plot_step_timer_counter(record):
    names = ["0-0.01","0.01-0.1", "0.1-1","1-10", "10-100","100+"]
    count = [0,0,0,0,0,0]
    for i in record:
        if i<0.01:
            count[0]+=1
        elif i<0.1:
            count[1]+=1
        elif i< 1:
            count[2]+=1
        elif i < 10:
            count[3]+=1
        elif i < 100:
            count[4]+=1
        else:
            count[5]+=0
    print(names)
    print(count)

    plt.bar(names, count)
    plt.suptitle("Step Timer Distribution")
    plt.xlabel("step time(seconds)")
    plt.ylabel("number of steps")
    plt.yscale("log")
    plt.show()
