import time

import numpy as np
from system.controller.explorationPhase import compute_exploration_goal_vector
from system.controller.navigationPhase import compute_navigation_goal_vector
from plotting.plotResults import *
from plotting.plotThesis import *
import matplotlib.animation as animation
import os
import matplotlib.pyplot as plt
from numba import jit

class Bio_Model:

    def __init__(self,trag_coding = "Full_Exploration", gc_network = None, env = None,pc_network = None, cognitive_map = None, nr_plots = 5,nr_steps = 8000, goal_idx = 255, bc_enabled = False, video= False, spike_detector = None,pod_network = None,construct_new_cognitive_map = False, nr_steps_exploration=0, fps= 5,step=1):
        self.trag_coding = trag_coding
        self. video = video
        self.fps = fps
        self.step = step
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
        self.goal_idx = goal_idx
        self.nr_steps = nr_steps
        self.nr_plots = nr_plots
        [self.fig, self.f_gc, self.f_t, self.f_mon] = layout_video()
        self.anim =None
        self.start_timer=None
        self.end_timer=None
        self.step_time_record=[]


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
            step_timer_start=time.time()
            # compute goal vector
            exploration_phase = True if i < self.nr_steps_exploration else False
            # compute the goal_vector from rodent to goal in global coordinate system
            if exploration_phase:
                compute_exploration_goal_vector(self.env, i, trag_coding = self.trag_coding)
            else:
                compute_navigation_goal_vector(self.gc_network, self.pc_network, self.cognitive_map, i - self.nr_steps_exploration, self.env,
                                               pod=self.pod_network, spike_detector=self.spike_detector, model=self.vector_model)
            self.goal_vector_array.append(self.env.goal_vector)

            # compute velocity vector
            self.env.compute_movement(self.gc_network, self.pc_network, self.cognitive_map, exploration_phase=exploration_phase)
            xy_speed = self.env.xy_speeds[-1]
            #print(xy_speed)

            # grid cell network track movement

            self.gc_network.track_movement(xy_speed)
            #if i % 1000 == 0:
            #    self.plot_GC_Projection_2(self.gc_network)
            #    print("Step: ", i," Current tuned direction is: ", self.check_tuned_direction_vector(self.gc_network))

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
            [firing_values, created_new_pc, PC_idx, neighbouring_PC] = self.pc_network.track_movement(self.gc_network.gc_modules,
                                                                                reward_first_found,
                                                                                generate_new_PC=self.construct_new_cognitive_map)

            #if i % 20 == 0 and i != 0:
                #print("current position: ", self.env.xy_coordinates[-1]," current firing values: ", firing_values)
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
                for neigh in neighbouring_PC:
                    if neigh not in self.env.visited_PCs:
                        self.env.visited_PCs.append(neigh)
                print("the visited PCs are: ", self.env.visited_PCs)

            ###changes by Haoyang Sun - end

            # cognitive map track pc firing
            self.cognitive_map.track_movement(firing_values, created_new_pc, reward)
            ###changes by Haoyang Sun-start
            ##if the robot has reached the goal, end the simulation
            if self.goal_idx == PC_idx:#in self.env.visited_PCs:
                break
            ###changes by Haoyang Sun-end
            # plot or print intermediate update in console
            if not self.video and i % int(self.nr_steps / self.nr_plots) == 0:
                progress_str = "Progress: " + str(int(i * 100 / self.nr_steps)) + "%"
                print(progress_str)
                # plotCurrentAndTarget(gc_network.gc_modules)
            step_timer_end = time.time()
            self.step_time_record.append(step_timer_end-step_timer_start)

        # simulated steps until next frame
        if self.video:
            # export current state as frame
            exploration_phase = True if frame < self.nr_steps_exploration else False
            plot_current_state(self.env, self.gc_network.gc_modules, self.f_gc, self.f_t, self.f_mon,
                               pc_network=self.pc_network, cognitive_map=self.cognitive_map,
                               exploration_phase=exploration_phase, goal_vector=self.goal_vector_array[-1])
            progress_str = "Progress: " + str(int((frame * 100) / self.nr_steps)) + "% | Current video is: " + str(
                frame * self.dt) + "s long"
            print(progress_str)

    def make_animation(self):
        frames = np.arange(0, self.nr_steps, self.step)
        self.anim = animation.FuncAnimation(self.fig, func=self.animation_frame, frames=frames, interval=1 / self.fps, blit=False)

        # Finished simulation

        # Export video
        directory = "videos/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        f = "videos/animation.mp4"
        video_writer = animation.FFMpegWriter(fps=self.fps)
        self.anim.save(f, writer=video_writer)
        self.env.end_simulation()

    def plot_GC_Projection(self):

        new_dim = self.gc_network.n
        fig, axs = plt.subplots(2, len(self.gc_network.gc_modules))
        for axis in range(2):
            proj_s_vectors = []
            for i, gc in enumerate(self.gc_network.gc_modules):
                s = gc.get_s()
                if not gc.CUDA:
                    s = np.reshape(s, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
                axs[axis][i].set_title("Nr " + str(i)+" Dir: " + str(gc.tuned_direction) + " on axis " + str(axis))
                pro = np.sum(s, axis=axis)
                axs[axis][i].barh(range(new_dim), pro)
                print(i, "  ", axis, "   ", np.sum(pro))

        plt.show()

    def plot_GC_Projection_2(self, gc_network):
        fig, axs = plt.subplots(2, len(gc_network.gc_modules))
        new_dim = gc_network.n
        for axis in range(2):
            proj_s_vectors = []
            for i, gc in enumerate(gc_network.gc_modules):
                s = np.reshape(gc.get_s(virtual=False), (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
                axs[axis][i].set_title(
                    "neural sheet Nr. " + str(i) + " on axis " + str(axis))
                pro = np.sum(s, axis=axis)
                axs[axis][i].barh(range(new_dim), pro)
                # print(i, "  ", axis, "   ", np.sum(pro))
        plt.show()

    def check_tuned_direction_vector(self,gc_network):
        direction_vector = []
        new_dim = gc_network.n
        for gc in gc_network.gc_modules:
            s = gc.get_s(virtual=False)
            filter = np.where(s > 0.1, 1.0, 0.0)
            s = np.multiply(s, filter)
            s = np.reshape(s, (new_dim, new_dim))

            # s_filtered = np.where(s>0.1, 1, 0)
            decided = False
            for axis in range(2):
                dir = 'x' if axis == 0 else 'y'
                pro = np.sum(s, axis=axis)
                # print("current direction is ", dir, " argmin is ", np.amin(pro))
                if np.amin(pro) == 0 and not decided:
                    direction_vector.append(dir)
                    decided = True
        return direction_vector
