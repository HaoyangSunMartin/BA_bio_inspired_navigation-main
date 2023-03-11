from system.controller.pybulletEnv import PybulletEnvironment
from plotting.plotResults import *
from plotting.plotThesis import *
from system.original_bio_model.gridcellModel import GridCellNetwork
from system.original_bio_model.gridcellModel_CUDA import GridCellNetwork_CUDA

from system.original_bio_model.blockcellModel import BlockCellList
from system.decoder.phaseOffsetDetector import PhaseOffsetDetectorNetwork
from system.decoder.spikeDetection import SpikeDetector
from system.original_bio_model.placecellModel import PlaceCellNetwork
from system.original_bio_model.cognitivemapModel import CognitiveMapNetwork

from system.controller.explorationPhase import compute_exploration_goal_vector
from system.controller.explorationPhase import get_exploration_trajectory
from system.controller.navigationPhase import compute_navigation_goal_vector
from system.controller.navigationPhase import conventional_method_A_star
from system.controller.navigationPhase import get_trajectory_from_place_cell
from system.conventional_methods.D_star_Lite_Test import Test


from system.simple_bio_model.model_interface_SIMPLE import Bio_Model_SIMPLE
from system.original_bio_model.model_interface import Bio_Model

import os
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import system.helper as helper
import time

#####Testing Ground For Python-start


generate = False


D_Star_Lite = False

if D_Star_Lite:
    test = Test(env_coding="", prior_knowledge_encoding= "", connectivity_style= "", interactive=False)
    test.play()



#####Testing Ground For Python-end
randomized_init = False
trag_coding = "Full_Exploration"
### changes by Haoyang Sun- Start
print(helper.timer4LinearLookAhead)
### changes by Haoyang Sun- End

mpl.rcParams['animation.ffmpeg_path'] = "ffmpeg/ffmpeg"

SIMPLE = True
Conventional = None

visualize = False
from_data = True
use_CUDA = False
bc_enabled =False
nr_steps = 8000#15000  # 8000 for decoder test, 15000 for maze exploration, 8000 for maze navigation
#nr_steps_exploration = nr_steps  # 3500 for decoder test, nr_steps for maze exploration, 0 for maze navigation
nr_steps_exploration = 0

construct_new_cognitive_map = False

conventional = False

goal_idx = 26

env_coding = "plane_doors"#doors_option = "plane_doors"  # "plane" for default, "plane_doors", "plane_doors_individual"
            #doors_option = "plane_doors"  "plane_doors_1" "plane_doors_2" "plane_doors_3" "plane_doors_4" "plane_doors_5c_3o"
            # "plane" for default, "plane_doors", "plane_doors_individual"
            #
center_block = False

if generate:
    from_data=False
    nr_steps=15000
    nr_steps_exploration=nr_steps
    construct_new_cognitive_map=True
    goal_idx = 50
    env_coding = "plane"



if SIMPLE:
    model = Bio_Model_SIMPLE()
else:
    model = Bio_Model()

# set time-step size of simulation
dt = 1e-2  # in seconds, use 1e-2 for sufficient results

# initialize Grid Cell Model
M = 7  # 6 for default, number of modules
n = 40  # 40 for default, size of sheet -> nr of neurons is squared
gmin = 0.2  # 0.2 for default, maximum arena size, 0.5 -> ~10m | 0.05 -> ~105m
gmax = 2.4 # 2.4 for default, determines resolution, dont pick to high (>2.4 at speed = 0.5m/s)
# note that if gc modules are created from data n and M are overwritten
if use_CUDA:
    gc_network = GridCellNetwork_CUDA(n, M, dt, gmin, gmax=gmax, from_data=from_data)
else:
    gc_network = GridCellNetwork(n, M, dt, gmin, gmax=gmax, from_data=from_data, randomized_init = randomized_init)


# initialize Grid Cell decoder and Pybullet Environment
#visualize = True  # set this to True if you want a simulation window to open (requires computational power)
###changes by HAOYANG SUN-start
visualize = visualize
###changes by HAOYANG SUN-end
env_model = "linear_sunburst"  # "plane" for default, "single_line_traversal", "linear_sunburst"
vector_model = "linear_lookahead"  # "linear_lookahead" for default, "phase_offset_detector", "spike_detection"
env_coding = env_coding

pod_network = PhaseOffsetDetectorNetwork(16, 9, n) if vector_model == "phase_offset_detector" else None
spike_detector = SpikeDetector() if vector_model == "spike_detection" else None

env = PybulletEnvironment(visualize, env_model, dt, pod=pod_network, doors_option=env_coding, center_block=center_block)

#from_data = False  # False for default, set to True if you want to load cognitive map data to the model
###changes by HAOYANG SUN-start

###changes by HAOYANG SUN-end

# initialize Place Cell Model
pc_network = PlaceCellNetwork(from_data=from_data, CUDA=use_CUDA)
# initialize Cognitive Map Model
cognitive_map = CognitiveMapNetwork(dt, from_data=from_data, CUDA=use_CUDA)

###
#print(pc_network.place_cells[goal_idx].compute_firing_2x(pc_network.place_cells[goal_idx].gc_connections, axis=0))

###
#print the current GC-network Orientation and the firing ralation according to projective distance

if bc_enabled:
    bc_list = BlockCellList(from_data=from_data)

if from_data:
    idx = np.argmax(cognitive_map.reward_cells)
    gc_network.set_as_target_state(pc_network.place_cells[idx].gc_connections)

###changes by Haoyang Sun-start
#plots before simulation
if conventional:
    print("the cognitive map has in total:", len(pc_network.place_cells), " place cells")
    pc_path = conventional_method_A_star(pc_network, cognitive_map, env, 29, 0)
    conven_tra = get_trajectory_from_place_cell(pc_path, pc_network)
    conven_dist = calculate_trajectory_distance(conven_tra)
    print("using the conventional approach, the pc_path is:", pc_path)
    print("using the conventional approach, the traveled distance is:", conven_dist)
    plot_trajectory_on_map(conven_tra, env, cognitive_map, pc_network, [0, 0], "convention_trajectory_map",
                           env_coding=env_coding,center_block=center_block)
if construct_new_cognitive_map:
    plot_trajectory_on_map(get_exploration_trajectory(trag_coding = trag_coding), env, cognitive_map, pc_network, [0, 0], "convention_trajectory_map",
                           env_coding=env_coding,center_block=center_block)
plot_cognitive_map(env,cognitive_map,pc_network,[0,0],"initial_cognitive_map",env_coding=env_coding,center_block=center_block)
###changes by Haoyang Sun-end

# run simulation
nr_steps = nr_steps #15000  # 8000 for decoder test, 15000 for maze exploration, 8000 for maze navigation
#nr_steps_exploration = nr_steps  # 3500 for decoder test, nr_steps for maze exploration, 0 for maze navigation
###changes by HAOYANG SUN-start
nr_steps_exploration = nr_steps_exploration
###changes by HAOYANG SUN-end
nr_plots = 5  # allows to plot during the run of a simulation
nr_trials = 0  # 1 for default, 50 for decoder test, 1 for maze exploration

# Configure video if you want to export one
#video = False  # False for default, set to True if you want to create a video of the run
###changes by HAOYANG SUN-start
video = False
###changes by HAOYANG SUN-end
fps = 5  # number of frames per s
step = int((1 / fps) / dt)  # after how many simulation steps a new frame is saved

if video:
    [fig, f_gc, f_t, f_mon] = layout_video()
else:
    fig = None

plot_matching_vectors = False  # False for default, True if you want to match spikes in the grid cel spiking plots


model = Bio_Model(trag_coding = trag_coding, construct_new_cognitive_map=construct_new_cognitive_map,goal_idx=goal_idx,nr_steps_exploration=nr_steps_exploration,nr_steps=nr_steps,video=video,gc_network=gc_network, pc_network = pc_network, cognitive_map = cognitive_map,env=env)


if video:
    # initialize video and call simulation function within
    frames = np.arange(0, nr_steps, step)
    anim = animation.FuncAnimation(fig, func=model.animation_frame, frames=frames, interval=1 / fps, blit=False)

    # Finished simulation

    # Export video
    directory = "videos/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = "videos/animation.mp4"
    video_writer = animation.FFMpegWriter(fps=fps)
    anim.save(f, writer=video_writer)
    env.end_simulation()
else:
    # manually call simulation function

    ###changed by Haoyang Sun--Start
    startTimer = time.time()
    ###changed by Haoyang Sun--End

    model.animation_frame(nr_steps)

    ###changed by Haoyang Sun--Start
    endTimer = time.time()
    totalTimer = (endTimer - startTimer)
    print("the total running time is:")
    print(totalTimer)
    print("linear look ahead takes:")
    print(helper.timer4LinearLookAhead)
    print("linear look ahead takes up(%):")
    print((helper.timer4LinearLookAhead*100/totalTimer))
    ###changed by Haoyang Sun--End

    # Finished simulation

    # Plot last state
    ###cognitive_map_plot(pc_network, cognitive_map, environment=env_model)
    cognitive_map_plot(pc_network, cognitive_map, env_coding=env_coding)
    # Save place network and cognitive map to reload it later
    pc_network.save_pc_network()  # provide filename="_navigation" to avoid overwriting the exploration phase
    cognitive_map.save_cognitive_map()  # provide filename="_navigation" to avoid overwriting the exploration phase

    ###changes by Haoyang Sun-start
    dist = calculate_trajectory_distance(env.xy_coordinates)
    print("the distance traveled is: ", dist)
    plot_trajectory_on_map(env.xy_coordinates, env, cognitive_map, pc_network, [0, 0], "convention_trajectory_map",
                           env_coding=env_coding)

    # plot_cognitive_map(env, cognitive_map, pc_network, [0, 0], "initial_cognitive_map")

    ###changes by Haoyang Sun-end
    #bc_list.save_bc_list()

    # Calculate the distance between goal and actual end position (only relevant for navigation phase)
    error = np.linalg.norm((env.xy_coordinates[-1] + env.goal_vector) - env.goal_location)
    env.end_simulation()  # disconnect pybullet

    # Data to save to perform analysis later on
    error_array = [error]
    gc_array = [gc_network.consolidate_gc_spiking()]
    position_array = [env.xy_coordinates]
    vector_array = [model.goal_vector_array]


"""
    progress_str = "Progress: " + str(int(1 * 100 / nr_trials)) + "% | Latest error: " + str(error)
    print(progress_str)

    # for the decoder test several trials are performed one after each other
    for i in range(1, nr_trials):
        gc_network.load_initialized_network("s_vectors_initialized.npy")
        pc_network = PlaceCellNetwork()
        cognitive_map = CognitiveMapNetwork(dt)
        env = PybulletEnvironment(visualize, env_model, dt)

        goal_vector_array = [np.array([0, 0])]

        model.animation_frame(nr_steps)
        error = np.linalg.norm((env.xy_coordinates[-1] + env.goal_vector) - env.goal_location)

        error_array.append(error)
        gc_array.append(gc_network.consolidate_gc_spiking())
        position_array.append(env.xy_coordinates)
        vector_array.append(goal_vector_array)

        env.end_simulation()

        progress_str = "Progress: " + str(int((i + 1) * 100 / nr_trials)) + "% | Latest error: " + str(error)
        print(progress_str)

    # Directly plot and print the errors (distance between goal and actual end position)
    error_plot(error_array)
    print(error_array)


    
    # Save the data of all trials in a dedicated folder
    directory = "experiments/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save("experiments/error_array", error_array)
    np.save("experiments/gc_array", gc_array)
    np.save("experiments/position_array", position_array)
    np.save("experiments/vectors_array", vector_array)

"""