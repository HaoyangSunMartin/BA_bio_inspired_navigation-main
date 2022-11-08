import pybullet as p
import time
import os
import pybullet_data
import numpy as np

from system.helper import compute_angle

from system.controller.navigationPhase import pick_intermediate_goal_vector, find_new_goal_vector


class PybulletEnvironment:
    """This class deals with everything pybullet or environment (obstacles) related"""
    def __init__(self, visualize, env_model, dt, pod=None, doors_option = "plane"):
        self.visualize = visualize  # to open JAVA application
        self.env_model = env_model  # string specifying env_model
        self.pod = pod  # if Phase Offset detectors are used

        if self.visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        base_position = [0, 0.05, 0.02]  # [0, 0.05, 0.02] ensures that it actually starts at origin
        arena_size = 15  # circular arena size with radius r
        goal_location = None
        ##changes of HAOYANG_SUN:
        #max_speed = 5.5  # determines speed at which agent travels: max_speed = 5.5 -> actual speed of ~0.5 m/s
        max_speed = 5.5
        if env_model == "plus":
            p.loadURDF("p3dx/plane/plane.urdf")
        elif env_model == "obstacle":
            p.loadURDF("environment/obstacle_map/plane.urdf")
        elif env_model == "linear_sunburst":
            ###Changes by Haoyang Sun - START
            #doors_option = "plane_doors"  # "plane" for default, "plane_doors", "plane_doors_individual"
            #doors_option = "plane_doors"  "plane_doors_1" "plane_doors_2" "plane_doors_3" "plane_doors_4" "plane_doors_5c_3o"
            # "plane" for default, "plane_doors", "plane_doors_individual"
            #doors_option = "plane"
            ###Changes by Haoyang Sun - END
            p.loadURDF("environment/linear_sunburst_map/" + doors_option + ".urdf")
            base_position = [5.5, 0.55, 0.02]
            arena_size = 15
            goal_location = np.array([1.5, 10])
            ##changes of HAOYANG_SUN:
            #max_speed = 6
            max_speed = 6
        else:
            urdfRootPath = pybullet_data.getDataPath()
            p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"))

        orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])  # faces North

        self.carID = p.loadURDF("p3dx/urdf/pioneer3dx.urdf", basePosition=base_position, baseOrientation=orientation)

        p.setGravity(0, 0, -9.81)

        self.dt = dt
        p.setTimeStep(self.dt)

        self.xy_coordinates = []  # keeps track of agent's coordinates at each time step
        self.orientation_angle = []  # keeps track of agent's orientation at each time step
        self.xy_speeds = []  # keeps track of agent's speed (vector) at each time step
        self.speeds = []  # keeps track of agent's speed (value) at each time step
        self.save_position_and_speed()  # save initial configuration

        # Set goal location to preset location or current position if none was specified
        self.goal_location = goal_location if goal_location is not None else self.xy_coordinates[0]

        self.max_speed = max_speed
        self.arena_size = arena_size
        self.goal = np.array([0, 0]) # used for navigation (eg. sub goals)


        self.goal_vector_original = np.array([1, 1])  # egocentric goal vector after last recalculation
        self.goal_vector = np.array([0, 0])  # egocentric goal vector after last update

        self.goal_idx = 0  # pc_idx of goal

        ### changes by Haoyang Sun - start
        ##this list records all the visited PCs, which will be excluded during LLA
        self.visited_PCs = []
        ### changes by Haoyang Sun - end

        self.turning = False  # agent state, used for controller

        self.num_ray_dir = 16  # number of direction to check for obstacles for
        self.num_travel_dir = 4  # valid traveling directions, 4 -> [E, N, W, S]
        self.directions = np.empty(self.num_ray_dir, dtype=bool)  # array keeping track which directions are blocked
        self.topology_based = False  # agent state, used for controller

    def compute_movement(self, gc_network, pc_network, cognitive_map, exploration_phase=True):
        """Compute and set motor gains of agents. Simulate the movement with py-bullet"""

        gains = self.avoid_obstacles(gc_network, pc_network, cognitive_map, exploration_phase)

        self.change_speed(gains)
        p.stepSimulation()

        self.save_position_and_speed()
        if self.visualize:
            time.sleep(self.dt/5)

    def change_speed(self, gains):
        p.setJointMotorControlArray(bodyUniqueId=self.carID,
                                    jointIndices=[4, 6],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=gains,
                                    forces=[10, 10])

    ###changes by Haoyang Sun - start
    #def goal_visited(self):
    #    self.visited_PCs.append(self.goal_idx)
    #    print("visited PCs are: ",self.visited_PCs)
    ###changes by Haoyang Sun - end


    #when called, this function saves the current position, current orientation(abosolute Euler Term in 2D plane)
    #xy_velocity and absolute velocity in respective variables
    def save_position_and_speed(self):
        [position, angle] = p.getBasePositionAndOrientation(self.carID)

        angle = p.getEulerFromQuaternion(angle)
        #the angle is in the absolute abgle in the 2D plane
        self.xy_coordinates.append(np.array([position[0], position[1]]))
        self.orientation_angle.append(angle[2])

        [linear_v, _] = p.getBaseVelocity(self.carID)
        self.xy_speeds.append([linear_v[0], linear_v[1]])
        self.speeds.append(np.linalg.norm([linear_v[0], linear_v[1]]))

    #basing on the goal-vector, the robot computes the gain(next move):
        # if the robot is close to the goal, the robot stops moving
        # if the difference in direction is huge, then the robot turns to the goal
        # otherwise the robot slightly adjust its direction relative to the goal
    #this function returns the velocity of the left and right motor respectively
    def compute_gains(self):
        """Calculates motor gains based on heading and goal vector direction"""
        current_angle = self.orientation_angle[-1]
        current_heading = [np.cos(current_angle), np.sin(current_angle)]
        diff_angle = compute_angle(current_heading, self.goal_vector) / np.pi

        gain = min(np.linalg.norm(self.goal_vector) * 5, 1)

        # If close to the goal do not move
        if gain < 0.5:
            gain = 0

        # If large difference in heading, do an actual turn
        if abs(diff_angle) > 0.05 and gain > 0:
            max_speed = self.max_speed / 2
            direction = np.sign(diff_angle)
            if direction > 0:
                v_left = max_speed * gain * -1
                v_right = max_speed * gain
            else:
                v_left = max_speed * gain
                v_right = max_speed * gain * -1
        else:
            # Otherwise only adjust course slightly
            self.turning = False
            max_speed = self.max_speed
            v_left = max_speed * (1 - diff_angle * 2) * gain
            v_right = max_speed * (1 + diff_angle * 2) * gain

        return [v_left, v_right]

    def end_simulation(self):
        p.disconnect()

    def avoid_obstacles(self, gc_network, pc_network, cognitive_map, exploration_phase):
        """Main controller function, to check for obstacles and adjust course if needed."""
        ray_reference = p.getLinkState(self.carID, 0)[1]
        current_heading = p.getEulerFromQuaternion(ray_reference)[2]  # in radians
        goal_vector_angle = np.arctan2(self.goal_vector[1], self.goal_vector[0])
        angles = np.linspace(0, 2 * np.pi, num=self.num_ray_dir, endpoint=False)
        emergent_manuver = False
        # direction where we want to check for obstacles
        angles = np.append(angles, [goal_vector_angle, current_heading])

        ray_dist = self.ray_detection(angles)  # determine ray values
        changed = self.update_directions(ray_dist)  # check if an direction became unblocked

        #The decision of using which kind of navigation method(topology or vector based)
        # if all directions are free, use vector based navigation
        if np.all(self.directions) and self.topology_based:
            # Switch back to vector-based navigation if all directions are free
            self.topology_based = False
            find_new_goal_vector(gc_network, pc_network, cognitive_map, self,
                                 model="linear_lookahead", pod=None, spike_detector=None)

        minimum_dist = np.min(ray_dist)
        # if there is an obstacle very close to the robot(less than 0.3m), back off from it(distance 0.5) and then start topology based navigation
        if minimum_dist < 0.3:
            # Initiate back off maneuver
            idx = np.argmin(ray_dist)
            angle = angles[idx] + np.pi
            self.goal_vector = np.array([np.cos(angle), np.sin(angle)]) * 0.5
            print("obstacle too close, backing off\n")
            self.goal_vector_original = self.goal_vector

            self.topology_based = True
            emergent_manuver = True
        ###??? the handling here need to be checked more carefully

        if not exploration_phase:
            if self.topology_based or ray_dist[-1] < 0.6 or ray_dist[-2] < 0.6:
                # Approaching an obstacle in heading or goal vector direction, or topology based
                if not self.topology_based or changed:
                    # Switching to topology-based, or already topology-based but new direction became available
                    self.topology_based = True
                    pick_intermediate_goal_vector(gc_network, pc_network, cognitive_map, self)

        if not emergent_manuver:
            for i, dis in enumerate(ray_dist):
                if dis < 0.9:
                    blocked_angle = angle[i]
                    vec = np.array([np.cos(blocked_angle), np.sin(blocked_angle)]) * 0.9
                    self.create_block_cell(gc_network, pc_network, cognitive_map,vec)



        return self.compute_gains()



    def update_directions(self, ray_dist):
        """Check which of the directions are blocked and if one became unblocked"""

        changed = False
        directions = np.ones_like(self.directions)
        #print("direction shape:")
        #print(self.directions.shape)
        for idx in range(self.num_ray_dir):
            left = idx - 1 if idx - 1 >= 0 else self.num_ray_dir - 1  # determine left direction in circle (the direction left of the current one)
            right = idx + 1 if idx + 1 <= self.num_ray_dir - 1 else 0  # determine right direction in circle (the direction right of the current one)
            if ray_dist[idx] < 1.3 or ray_dist[left] < 0.9 or ray_dist[right] < 0.9:
                # If in direction an obstacle is nearby or in one of the neighbouring, then it is blocked
                directions[idx] = False
            if idx % self.num_travel_dir == 0 and directions[idx] and not self.directions[idx]:
                # One of the traveling directions became unblocked
                changed = True
        self.directions = directions
        return changed

    def ray_detection(self, angles):
        """Check for obstacles in defined directions."""

        p.removeAllUserDebugItems()

        ray_len = 2  # max_ray length to check for(horizon of the robot)

        ray_from = []
        ray_to = []

        ray_from_point = np.array(p.getLinkState(self.carID, 0)[0])
        ray_from_point[2] = ray_from_point[2] + 0.02

        for angle in angles:
            ray_from.append(ray_from_point)
            ray_to.append(np.array([
                np.cos(angle) * ray_len + ray_from_point[0],
                np.sin(angle) * ray_len + ray_from_point[1],
                ray_from_point[2]
            ]))

        ray_dist = np.empty_like(angles)
        results = p.rayTestBatch(ray_from, ray_to, numThreads=0) #perform a single ray-cast to find the intersection information of the first object hit
        for idx, result in enumerate(results):
            hit_object_uid = result[0]

            dist = ray_len
            if hit_object_uid != -1:
                hit_position = result[3]
                ###dist =  np.linalg.norm(hit_position - ray_from_point)
                ###changed by Haoyang Sun- Start
                dist = result[2]*2
                ###ray_dist[idx] = dist
                ###changed by Haoyang Sun- End
            ray_dist[idx] = dist

            # if dist < 1:
            #     p.addUserDebugLine(ray_from[idx], ray_to[idx], (1, 1, 1))

        return ray_dist


    def detect_obstacles(self):
        return self.directions

    ###changes by Haoyang Sun - start
    ##this function creates block cell with an allocentric position as input
    def create_block_cell(self, gc_network, pc_network, cognitive_map, vector):
        gc_network.reset_s_virtual()
        gc_network.track_movement(vector, virtual=True, dt_alternative=1)

        return


    ###changes by Haoyang Sun - end






