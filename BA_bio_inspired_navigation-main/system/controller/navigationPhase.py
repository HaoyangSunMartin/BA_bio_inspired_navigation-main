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

    env.goal_vector = perform_lookahead_directed(gc_network, pc_network, cognitive_map, env)
    env.goal_vector_original = env.goal_vector

    ###changes by Haoyang Sun--Start
    end = time.time()
    helper.timer4LinearLookAhead = helper.timer4LinearLookAhead + (end - start)
    ###changes by Haoyang Sun--End



