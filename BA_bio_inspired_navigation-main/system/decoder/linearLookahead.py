import numpy as np
from plotting.plotResults import export_linear_lookahead_video
from plotting.plotThesis import plot_sub_goal_localization


def perform_look_ahead_2x(gc_network, pc_network, cognitive_map, env, video=False, plotting=False, goal_pc_idx=None):
    """Performs a linear lookahead to find an offset in grid cell spiking in either x or y direction."""
    gc_network.reset_s_virtual()  # Resets virtual gc spiking to actual spiking

    dt = gc_network.dt * 10  # checks spiking only every nth step
    speed = 0.5  # match actual speed
    xy_speeds = np.array(([1, 0], [-1, 0], [0, 1], [0, -1])) * speed
    goal_spiking = {}  # "axis": {"reward_value", "idx_place_cell", "distance", "step"}

    max_distance = 1.1 * env.arena_size  # after this distance lookahead is aborted

    max_nr_steps = int(max_distance / (speed * dt))

    for idx, xy_speed in enumerate(xy_speeds):
        axis = int(idx / 2)  # either x or y

        reward_array = []  # save rewards during lookahead, to create lookahead video
        goal_found = None
        for i in range(max_nr_steps):
            ###plot = True if (plotting and i == 0 and idx % 2 == 0) else False  # for plotting purposes
            ###changed by Haoyang Sun--Start
            plot = False
            ###changed by Haoyang Sun--End

            # Compute reward firing
            if goal_pc_idx is None:
                # Check all pc cells and identify reward spiking
                firing_values = pc_network.compute_firing_values(gc_network.gc_modules, virtual=True, axis=axis, plot=plot)
                [reward, idx_place_cell] = cognitive_map.compute_reward_spiking(firing_values)
            else:
                # Only look for one specific place cell (goal location)
                s_vectors = gc_network.consolidate_gc_spiking(virtual=True)

                # computes projected pc firing
                firing = pc_network.place_cells[goal_pc_idx].compute_firing_2x(s_vectors, axis)

                # make sure that firing is strong enough
                reward = firing if firing > cognitive_map.active_threshold else 0
                idx_place_cell = goal_pc_idx

            reward_array.append(reward)

            distance = xy_speed[axis] * i * dt  # lookahead distance
            if axis not in goal_spiking or reward - goal_spiking[axis]["reward"] > 0:
                # First entrance or exceeds previous found value
                goal_spiking[axis] = {"reward": reward, "idx_place_cell": idx_place_cell,
                                      "distance": distance, "step": i}
                goal_found = i

            # Abort conditions to end lookahead earlier
            if axis in goal_spiking and i > 50 and reward < 0.85 * goal_spiking[axis]["reward"]\
                    and goal_spiking[axis]["reward"] > 0.9:
                # To make sure that looks sufficiently in all 4 directions add this above
                # and np.sign(goal_spiking[axis]["distance"]) == np.sign(distance) \
                break

            gc_network.track_movement(xy_speed, virtual=True, dt_alternative=dt)  # track virtual movement

            # if i % 20 == 0:
            #     print_str = "Lookahead progress| Direction " + str(idx) + "/4 " \
            #                 "| time-step " + str(i) + "/" + str(max_nr_steps) + \
            #                 "| total " + str(int(100 * (idx * max_nr_steps + i) / (4 * max_nr_steps))) + "%"
            #     print(print_str)

        if video is True:
            filename = "videos/linear_lookahead/linear_lookahead_" + str(idx) + ".mp4"
            export_linear_lookahead_video(gc_network, filename, xy_coordinates=env.xy_coordinates,
                                          reward_array=reward_array, goal_found=goal_found)
            print("Exported Video in direction:", idx)
        # elif goal_found is not None:
        #     [fig, f_gc, f_t, f_mon] = layout_video()
        #     plot_linear_lookahead(f_gc, f_t, f_mon, goal_found, gc_network, env.xy_coordinates, reward_array, goal_found)
        #     fig.show()

        gc_network.reset_s_virtual()  # reset after lookahead in a direction

    goal_vector = np.array([axis["distance"] for axis in goal_spiking.values()])  # consolidate goal vector from dict

    print("------ Goal localization at time-step: ", len(env.xy_coordinates) - 1)
    if len(goal_vector) != 2:
        # Something went wrong and no goal vector was found
        goal_vector = np.random.rand(2) * 0.5
        print("Unable to find a goal_vector", goal_spiking)
    else:
        print("Found goal vector", goal_vector, goal_spiking)

    filename = "_goal_" + str(len(env.xy_coordinates) - 1)

    ###plot_sub_goal_localization(env, cognitive_map, pc_network, goal_vector, filename)
    ###changed by Haoyang Sun--Start

    ###changed by Haoyang Sun--End
    return goal_vector


def perform_lookahead_directed(gc_network, pc_network, cognitive_map, env):
    """Performs a linear lookahead in a preset direction"""
    gc_network.reset_s_virtual()  # Resets virtual gc spiking to actual spiking

    dt = gc_network.dt * 40  # checks spiking only every nth step
    speed = 0.5  # lookahead speed, becomes unstable for large speeds
    #this can be the main reason causing the slow reaction

    angles = np.linspace(0, 2 * np.pi, num=env.num_ray_dir, endpoint=False)  # lookahead directions

    goal_spiking = {}  # "angle": {"reward_value", "idx_place_cell", "distance", "step"}

    max_distance = 0.5 * env.arena_size  # after this distance lookahead is aborted
    max_nr_steps = int(max_distance / (speed * dt))

    for idx, angle in enumerate(angles):

        # Check if lookahead direction is blocked
        if not env.directions[idx]:
            # If yes do not consider that direction
            goal_spiking[angle] = {"reward": -1, "idx_place_cell": -1,
                                   "distance": 0, "step": 0, "blocked": True}
            continue

        # Check if direction is one of the favored traveling directions
        if not idx % env.num_travel_dir == 0:
            # If no do not consider that direction
            goal_spiking[angle] = {"reward": -1, "idx_place_cell": -1,
                                   "distance": 0, "step": 0, "blocked": False}
            continue

        xy_speed = np.array([np.cos(angle), np.sin(angle)]) * speed  # lookahead velocity vector

        for i in range(max_nr_steps):
            firing_values = pc_network.compute_firing_values(gc_network.gc_modules, virtual=True)
            [reward, idx_place_cell] = cognitive_map.compute_reward_spiking(firing_values)  # highest reward spiking

            distance = np.linalg.norm(xy_speed * i * dt)  # lookahead distance traveled
            if angle not in goal_spiking or reward - goal_spiking[angle]["reward"] > 0:
                # First entrance or exceeds previous found value
                goal_spiking[angle] = {"reward": reward, "idx_place_cell": idx_place_cell,
                                       "distance": distance, "step": i, "blocked": False}

            # Abort conditions to end lookahead earlier
            if angle in goal_spiking and reward < 0.85 * goal_spiking[angle]["reward"] \
                    and goal_spiking[angle]["reward"] > 0.8 and i > 50:
                break

            gc_network.track_movement(xy_speed, virtual=True, dt_alternative=dt)  # track virtual movement

        gc_network.reset_s_virtual()  # reset after lookahead in a direction

    idx_angle = np.argmax([angle["reward"] for angle in goal_spiking.values()])  # determine most promising direction
    angle = list(goal_spiking.keys())[idx_angle]

    reward = goal_spiking[angle]["reward"]

    print("------ Sub goal localization at time-step: ", len(env.xy_coordinates) - 1)
    for alternative_angle in angles:
        # Print all angles that showed potential
        if goal_spiking[alternative_angle]["reward"] != -1:
            print(alternative_angle, goal_spiking[alternative_angle])

    #  If the agent is very close to the goal, it switches to vector-based navigation
    if reward > 0.9:
        env.topology_based = False
        goal_vector = perform_look_ahead_2x(gc_network, pc_network, cognitive_map, env,
                                            goal_pc_idx=goal_spiking[angle]["idx_place_cell"])
    elif reward >= 0:
        distance = goal_spiking[angle]["distance"]
        distance = np.maximum(distance, 0.5) if reward < 0.8 else distance
        goal_vector = np.array([np.cos(angle), np.sin(angle)]) * distance  # goal vector to travel along
        print("Choose goal spiking: ", goal_vector, angle, goal_spiking[angle])

    else:
        goal_vector = np.random.rand(2) * 0.5
        print("No goal vector found, trying random ", goal_vector)

    filename = "_subgoal_" + str(len(env.xy_coordinates) - 1)
    ###plot_sub_goal_localization(env, cognitive_map, pc_network, env.goal_vector, filename, idx_angle, goal_spiking)
    ###changed by Haoyang Sun--Start
    ###changed by Haoyang Sun--End
    return goal_vector
