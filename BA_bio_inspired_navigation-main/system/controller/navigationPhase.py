from system.decoder.linearLookahead import *
import numpy as np
import system.helper as helper
import time




def compute_navigation_goal_vector(gc_network, pc_network, cognitive_map, nr_steps, env,
                                   model="linear_lookahead", pod=None, spike_detector=None):
    """Computes the goal vector for the agent to travel to"""


    distance_to_goal = np.linalg.norm(env.goal_vector)  # current length of goal vector
    distance_to_goal_original = np.linalg.norm(env.goal_vector_original)  # length of goal vector at calculation

    update_fraction = 0.2 if model == "linear_lookahead" else 0.5  # how often the goal vector has to be recalculated
    if env.topology_based and distance_to_goal < 0.3:
        # Agent has reached sub goal in topology based navigation -> pick next goal
        pick_intermediate_goal_vector(gc_network, pc_network, cognitive_map, env)
    elif (not env.topology_based and distance_to_goal/distance_to_goal_original < update_fraction
          and distance_to_goal_original > 0.3) or nr_steps == 0:
        # Vector-based navigation and agent has traversed a large portion of the goal vector, it is recalculated
        find_new_goal_vector(gc_network, pc_network, cognitive_map, env,
                             model=model, pod=pod, spike_detector=spike_detector)
    else:
        # Otherwise vector is not recalculated but just updated according to traveling speed
        env.goal_vector = env.goal_vector - np.array(env.xy_speeds[-1]) * env.dt


def find_new_goal_vector(gc_network, pc_network, cognitive_map, env,
                         model="linear_lookahead", pod=None, spike_detector=None):
    """For Vector-based navigation, computes goal vector with one grid cell decoder"""

    # video = True if nr_steps == 0 else False
    # plot = True if nr_steps == 0 else False
    video = False
    plot = False

    if model == "spike_detection":
        vec_avg_overall = spike_detector.compute_direction_signal(gc_network.gc_modules)
        env.goal_vector = vec_avg_overall
    elif model == "phase_offset_detector" and pod is not None:
        env.goal_vector = pod.compute_goal_vector(gc_network.gc_modules)
    else:
        goal_pc_idx = np.argmax(cognitive_map.reward_cells)  # pick which pc to look for (reduces computational effort)


        start = time.time()
        env.goal_vector = perform_look_ahead_2x(gc_network, pc_network, cognitive_map, env,
                                                goal_pc_idx=goal_pc_idx, video=video, plotting=plot)
        end = time.time()
        helper.timer4LinearLookAhead = helper.timer4LinearLookAhead + (end - start)

    env.goal_vector_original = env.goal_vector


def pick_intermediate_goal_vector(gc_network, pc_network, cognitive_map, env):
    """For topology-based navigation, computes sub goal vector with directed linear lookahead"""

    # Alternative option to calculate sub goal vector with phase offset detectors
    # if env.pod is not None:
    # env.goal_vector = env.pod.compute_sub_goal_vector(gc_network, pc_network, cognitive_map, env, blocked_directions)
    # else:
    ###changes by Haoyang Sun--Start
    start = time.time()
    ###changes by Haoyang Sun--End


    ###env.goal_vector = perform_lookahead_directed(gc_network, pc_network, cognitive_map, env)
    ###env.goal_vector_original = env.goal_vector
    ###changes by Haoyang Sun--Start
    #video = False
    #plot = False
    #env.goal_vector = perform_look_ahead_2x(gc_network, pc_network, cognitive_map, env,
    #                                            goal_pc_idx=goal_pc_idx, video=video, plotting=plot)
    env.goal_vector = perform_lookahead_directed(gc_network, pc_network, cognitive_map, env)
    env.goal_vector_original = env.goal_vector
    ###changes by Haoyang Sun--End


    ###changes by Haoyang Sun--Start
    end = time.time()
    helper.timer4LinearLookAhead = helper.timer4LinearLookAhead + (end - start)
    ###changes by Haoyang Sun--End

 #find the cell that has the lowest value among remaining cells
def find_argmin_with_filter(filter, list):
    x = []
    for i in range(len(list)):
        if i in filter:
            x.append(list[i])
    minimum = min(x)

    for i in filter:
        if list[i] == minimum:
            return i


#apply Dijkstra Algorithms to the cognitive map, to find path from the goal to the current.
def conventional_method(pc_network, cognitive_map, env, goal,current):
    nr_cells = len(pc_network.place_cells)
    distance = [1000 for x in range(nr_cells)]
    distance[goal] = 0
    path = {}
    remaining = []
    for x in range(nr_cells):
        path[x] =[]
        remaining.append(x)
    cont = True
    path[goal] = [goal]
    while(cont):
        #find the cell that has the lowest value among remaining cells
        i = find_argmin_with_filter(remaining, distance)
        n_distance = distance[i]+1
        #expand the edges of this cell and update neighbouring cells
        for j, connection in enumerate(cognitive_map.topology_cells[i]):
            if connection == 1 and i != j and n_distance < distance[j]:
                distance[j] = n_distance
                path[j] = path[i].copy()
                path[j].insert(0,j)
        remaining.remove(i)
        if path[current] != []:
            cont = False
    return path[current]

def get_trajectory_from_place_cell(path , pc_network):
    xy_coordinates = []
    for next in path:
        xy_coordinates.insert(0,pc_network.place_cells[next].env_coordinates)
    return xy_coordinates

def get_current_pc():
    return
def get_goal_pc():
    return










