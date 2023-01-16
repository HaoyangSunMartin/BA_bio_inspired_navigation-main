import numpy as np
from system.controller.explorationPhase import compute_exploration_goal_vector
from system.controller.navigationPhase import compute_navigation_goal_vector
from plotting.plotResults import *
from plotting.plotThesis import *

class Bio_Model:

    def __init__(self, gc_network = None, env = None,pc_network = None, cognitive_map = None, nr_plots = 5,nr_steps = 8000, goal_idx = 255, bc_enabled = False, video= False, spike_detector = None,pod_network = None,construct_new_cognitive_map = False, nr_steps_exploration=0):
        self. video = video
        self.step = 1
        self.nr_steps_exploration = nr_steps_exploration
        self.env = env
        self.gc_network = gc_network
        self.pc_network = pc_network
        self.cognitive_map = cognitive_map
        self.construct_new_cognitive_map = construct_new_cognitive_map
        self.pod_network =pod_network
        self.spike_detector = spike_detector
        self.goal_vector_array = [np.array([0, 0])]
        self.bc_enabled = bc_enabled
        self.vector_model = "linear_lookahead"
        self.bc_list = None
        self.goal_idx = 255
        self.nr_steps = nr_steps
        self.nr_plots = nr_plots
        [self.fig, self.f_gc, self.f_t, self.f_mon] = layout_video()

    def animation_frame(self, frame):
        if self.video:
            # calculate how many simulations steps to do for current frame
            start = frame - self.step
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
            exploration_phase = True if i < self.nr_steps_exploration else False
            # compute the goal_vector from rodent to goal in global coordinate system
            if exploration_phase:
                compute_exploration_goal_vector(self.env, i)
            else:
                compute_navigation_goal_vector(self.gc_network, self.pc_network, self.cognitive_map, i - self.nr_steps_exploration, self.env,
                                               pod=self.pod_network, spike_detector=self.spike_detector, model=self.vector_model)
            self.goal_vector_array.append(self.env.goal_vector)

            # compute velocity vector
            self.env.compute_movement(self.gc_network, self.pc_network, self.cognitive_map, exploration_phase=exploration_phase)
            xy_speed = self.env.xy_speeds[-1]

            # grid cell network track movement
            self.gc_network.track_movement(xy_speed)

            # place cell network track gc firing
            goal_distance = np.linalg.norm(self.env.xy_coordinates[-1] - self.env.goal_location)

            # during exploration Phase, create the goal place cell
            # print("current goal location is: ", env.goal_location)
            reward = 1 if goal_distance < 0.1 else 0
            reward_first_found = False
            if reward == 1 and (len(self.cognitive_map.reward_cells) == 0 or np.max(self.cognitive_map.reward_cells) != 1):
                reward_first_found = True
                self.gc_network.set_current_as_target_state()

            # update the PC map
            [firing_values, created_new_pc, PC_idx] = self.pc_network.track_movement(self.gc_network.gc_modules,
                                                                                reward_first_found,
                                                                                generate_new_PC=self.construct_new_cognitive_map)

            if i % 20 == 0 and i != 0:
                print("current position: ", self.env.xy_coordinates[-1]," current firing values: ", firing_values)
            # Block Cell List Track Movements:
            if self.bc_enabled:
                if i % 15 == 0:
                    [created_new_bc, bc_coordinate] = self.bc_list.track_movement(self.gc_network, self.env)

            if created_new_pc:
                print("creat pc Nr. ", len(self.pc_network.place_cells))
                #self.gc_network.plot_modules(i)
                self.pc_network.place_cells[-1].env_coordinates = np.array(self.env.xy_coordinates[-1])

            ###changes by Haoyang Sun - start
            if len(self.env.visited_PCs) == 0 or self.env.visited_PCs[-1] != PC_idx:
                self.env.visited_PCs.append(PC_idx)
                print("the visited PCs are: ", self.env.visited_PCs)

            ###changes by Haoyang Sun - end

            # cognitive map track pc firing
            self.cognitive_map.track_movement(firing_values, created_new_pc, reward)
            ###changes by Haoyang Sun-start
            ##if the robot has reached the goal, end the simulation
            if self.goal_idx in self.env.visited_PCs:
                break
            ###changes by Haoyang Sun-end
            # plot or print intermediate update in console
            if not self.video and i % int(self.nr_steps / self.nr_plots) == 0:
                progress_str = "Progress: " + str(int(i * 100 / self.nr_steps)) + "%"
                print(progress_str)
                # plotCurrentAndTarget(gc_network.gc_modules)

        # simulated steps until next frame
        if self.video:
            # export current state as frame
            exploration_phase = True if frame < self.nr_steps_exploration else False
            plot_current_state(self.env, self.gc_network.gc_modules, f_gc, f_t, f_mon,
                               pc_network=self.pc_network, cognitive_map=self.cognitive_map,
                               exploration_phase=exploration_phase, goal_vector=self.goal_vector_array[-1])
            progress_str = "Progress: " + str(int((frame * 100) / self.nr_steps)) + "% | Current video is: " + str(
                frame * dt) + "s long"
            print(progress_str)