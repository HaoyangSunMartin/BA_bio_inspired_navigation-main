
from system.controller.pybulletEnv import PybulletEnvironment
from plotting.plotResults import *
from plotting.plotThesis import *
from system.bio_model.gridcellModel import GridCellNetwork
from system.decoder.phaseOffsetDetector import PhaseOffsetDetectorNetwork
from system.decoder.spikeDetection import SpikeDetector
from system.bio_model.placecellModel import PlaceCellNetwork
from system.bio_model.cognitivemapModel import CognitiveMapNetwork

from system.controller.explorationPhase import compute_exploration_goal_vector
from system.controller.navigationPhase import compute_navigation_goal_vector
from system.controller.navigationPhase import conventional_method
from system.controller.navigationPhase import get_trajectory_from_place_cell

import os
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import system.helper as helper
import time


### changes by Haoyang Sun- Start
print(helper.timer4LinearLookAhead)
### changes by Haoyang Sun- End

mpl.rcParams['animation.ffmpeg_path'] = "ffmpeg/ffmpeg"

from_data = True
# set time-step size of simulation
dt = 1e-2  # in seconds, use 1e-2 for sufficient results

# initialize Grid Cell Model
M = 6  # 6 for default, number of modules
n = 40  # 40 for default, size of sheet -> nr of neurons is squared
gmin = 0.2  # 0.2 for default, maximum arena size, 0.5 -> ~10m | 0.05 -> ~105m
gmax = 2.4  # 2.4 for default, determines resolution, dont pick to high (>2.4 at speed = 0.5m/s)
# note that if gc modules are created from data n and M are overwritten
gc_network = GridCellNetwork(n, M, dt, gmin, gmax=gmax, from_data=from_data)

# initialize Grid Cell decoder and Pybullet Environment
#visualize = True  # set this to True if you want a simulation window to open (requires computational power)
###changes by HAOYANG SUN-start
visualize = False
###changes by HAOYANG SUN-end
env_model = "linear_sunburst"  # "plane" for default, "single_line_traversal", "linear_sunburst"
vector_model = "linear_lookahead"  # "linear_lookahead" for default, "phase_offset_detector", "spike_detection"
env_coding = "plane_doors_5c_3o"

pod_network = PhaseOffsetDetectorNetwork(16, 9, n) if vector_model == "phase_offset_detector" else None
spike_detector = SpikeDetector() if vector_model == "spike_detection" else None

env = PybulletEnvironment(visualize, env_model, dt, pod=pod_network)

#from_data = False  # False for default, set to True if you want to load cognitive map data to the model
###changes by HAOYANG SUN-start

###changes by HAOYANG SUN-end

# initialize Place Cell Model
pc_network = PlaceCellNetwork(from_data=from_data)
# initialize Cognitive Map Model
cognitive_map = CognitiveMapNetwork(dt, from_data=from_data)

if from_data:
    idx = np.argmax(cognitive_map.reward_cells)
    gc_network.set_as_target_state(pc_network.place_cells[idx].gc_connections)
###changes by Haoyang Sun-start
#plots before simulation
print("the cognitive map has in total:", len(pc_network.place_cells), " place cells")
pc_path = conventional_method(pc_network, cognitive_map, env, 29, 0 )
conven_tra= get_trajectory_from_place_cell(pc_path, pc_network)
conven_dist=calculate_trajectory_distance(conven_tra)
print("using the conventional approach, the pc_path is:", pc_path)

print("using the conventional approach, the traveled distance is:", conven_dist)

plot_trajectory_on_map(conven_tra,env,cognitive_map,pc_network,[0,0],"convention_trajectory_map",env_coding=env_coding)
plot_cognitive_map(env,cognitive_map,pc_network,[0,0],"initial_cognitive_map",env_coding=env_coding)
###changes by Haoyang Sun-end
# run simulation
nr_steps = 7000 #15000  # 8000 for decoder test, 15000 for maze exploration, 8000 for maze navigation
#nr_steps_exploration = nr_steps  # 3500 for decoder test, nr_steps for maze exploration, 0 for maze navigation
###changes by HAOYANG SUN-start
nr_steps_exploration = 0
###changes by HAOYANG SUN-end
nr_plots = 5  # allows to plot during the run of a simulation
nr_trials = 1  # 1 for default, 50 for decoder test, 1 for maze exploration

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

# Save across frames
goal_vector_array = [np.array([0, 0])]  # array to save the calculated goal vector

### Start of Animation_Frame function
# this function performs the simulation steps and is called by video creator or manually
def animation_frame(frame):
    if video:
        # calculate how many simulations steps to do for current frame
        start = frame - step
        end = frame
        if start < 0:
            start = 0
    else:
        # run through all simulation steps as no frames have to be exported
        start = 0
        end = frame

    for i in range(start, end):

        # perform one simulation step

        # compute goal vector
        exploration_phase = True if i < nr_steps_exploration else False
        # compute the goal_vector from rodent to goal in global coordinate system
        if exploration_phase:
            compute_exploration_goal_vector(env, i)
        else:
            compute_navigation_goal_vector(gc_network, pc_network, cognitive_map, i - nr_steps_exploration, env,
                                           pod=pod_network, spike_detector=spike_detector, model=vector_model)
        goal_vector_array.append(env.goal_vector)


        # compute velocity vector
        env.compute_movement(gc_network, pc_network, cognitive_map, exploration_phase=exploration_phase)
        xy_speed = env.xy_speeds[-1]

        # grid cell network track movement
        gc_network.track_movement(xy_speed)

        # place cell network track gc firing
        goal_distance = np.linalg.norm(env.xy_coordinates[-1] - env.goal_location)
        #print("current goal location is: ", env.goal_location)
        reward = 1 if goal_distance < 0.1 else 0
        reward_first_found = False
        if reward == 1 and (len(cognitive_map.reward_cells) == 0 or np.max(cognitive_map.reward_cells) != 1):
            reward_first_found = True
            gc_network.set_current_as_target_state()


        [firing_values, created_new_pc, PC_idx] = pc_network.track_movement(gc_network.gc_modules, reward_first_found, generate_new_PC=False)



        if created_new_pc:
            pc_network.place_cells[-1].env_coordinates = np.array(env.xy_coordinates[-1])

        ###changes by Haoyang Sun - start
        if len(env.visited_PCs) ==0 or env.visited_PCs[-1] != PC_idx:
            env.visited_PCs.append(PC_idx)
            print("the visited PCs are: ", env.visited_PCs)
        # cognitive map track pc firing
        cognitive_map.track_movement(firing_values, created_new_pc, reward)

        # plot or print intermediate update in console
        if not video and i % int(nr_steps / nr_plots) == 0:
            progress_str = "Progress: " + str(int(i * 100 / nr_steps)) + "%"
            print(progress_str)
            # plotCurrentAndTarget(gc_network.gc_modules)

    # simulated steps until next frame
    if video:
        # export current state as frame
        exploration_phase = True if frame < nr_steps_exploration else False
        plot_current_state(env, gc_network.gc_modules, f_gc, f_t, f_mon,
                           pc_network=pc_network, cognitive_map=cognitive_map,
                           exploration_phase=exploration_phase, goal_vector=goal_vector_array[-1])
        progress_str = "Progress: " + str(int((frame * 100) / nr_steps)) + "% | Current video is: " + str(
            frame * dt) + "s long"
        print(progress_str)
### End of Animation_Frame function

if video:
    # initialize video and call simulation function within
    frames = np.arange(0, nr_steps, step)
    anim = animation.FuncAnimation(fig, func=animation_frame, frames=frames, interval=1 / fps, blit=False)

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

    animation_frame(nr_steps)

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

    # Calculate the distance between goal and actual end position (only relevant for navigation phase)
    error = np.linalg.norm((env.xy_coordinates[-1] + env.goal_vector) - env.goal_location)
    env.end_simulation()  # disconnect pybullet

    # Data to save to perform analysis later on
    error_array = [error]
    gc_array = [gc_network.consolidate_gc_spiking()]
    position_array = [env.xy_coordinates]
    vector_array = [goal_vector_array]

    progress_str = "Progress: " + str(int(1 * 100 / nr_trials)) + "% | Latest error: " + str(error)
    print(progress_str)

    # for the decoder test several trials are performed one after each other
    for i in range(1, nr_trials):
        gc_network.load_initialized_network("s_vectors_initialized.npy")
        pc_network = PlaceCellNetwork()
        cognitive_map = CognitiveMapNetwork(dt)
        env = PybulletEnvironment(visualize, env_model, dt)

        goal_vector_array = [np.array([0, 0])]

        animation_frame(nr_steps)
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


    ###changes by Haoyang Sun-start
    dist = calculate_trajectory_distance(env.xy_coordinates)
    print("the distance traveled is: ", dist)
    plot_trajectory_on_map(env.xy_coordinates, env, cognitive_map, pc_network, [0, 0], "convention_trajectory_map", env_coding=env_coding)


    #plot_cognitive_map(env, cognitive_map, pc_network, [0, 0], "initial_cognitive_map")

    ###changes by Haoyang Sun-end
    # Save the data of all trials in a dedicated folder
    directory = "experiments/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save("experiments/error_array", error_array)
    np.save("experiments/gc_array", gc_array)
    np.save("experiments/position_array", position_array)
    np.save("experiments/vectors_array", vector_array)

