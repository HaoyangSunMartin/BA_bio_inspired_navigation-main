from system.conventional_methods.gui import Animation, NoAnimation
from system.conventional_methods.d_star_lite import DStarLite,A_Star
from system.conventional_methods.grid import OccupancyGridMap, SLAM, get_environment, get_environment2, get_environment3


class Test:
    def __init__(self, ground_truth_env="plane_doors", prior_knowledge_env= "plane", interactive = True, scale=1, view_range=7):


        self.scale=scale
        view_range = view_range
        #this is the map during that navigation
        grid, start, goal = get_environment(env=ground_truth_env, scale=self.scale)
        # this is the map during the exploration
        grid2, _, _ = get_environment(env=prior_knowledge_env, scale=self.scale)


        x_dim = grid.shape[0]
        y_dim = grid.shape[1]

        self.map = OccupancyGridMap(x_dim, y_dim, own_map=grid)
        # print(map.occupancy_grid_map)
        # print(A_Star(map, s_goal=goal))

        # exit(0)
        if interactive:

            self.gui = Animation(title="D* Lite Path Planning",
                            width=int(7/self.scale),
                            height=int(7/self.scale),
                            margin=0,
                            x_dim=x_dim,
                            y_dim=y_dim,
                            start=start,
                            goal=goal,
                            viewing_range=view_range,
                            own_map=grid)
        else:
            self.gui = NoAnimation(title="D* Lite Path Planning",width=int(7/self.scale),height=int(7/self.scale),margin=0,x_dim=x_dim,y_dim=y_dim,start=start,goal=goal, viewing_range=view_range,own_map=grid)

        self.new_map = self.gui.world


        self.old_map = OccupancyGridMap(x_dim, y_dim, own_map=grid2)

        self.new_position = start
        self.last_position = start

        # new_observation = None
        # type = OBSTACLE

        # D* Lite (optimized)
        self.dstar = DStarLite(map=self.old_map,
                          s_start=start,
                          s_goal=goal)

        # SLAM to detect vertices
        self.slam = SLAM(map=self.new_map,
                    view_range=view_range)

    def play(self):
        # move and compute path
        path, g, rhs = self.dstar.move_and_replan(robot_position=self.new_position)
        nr_step = 0
        while not self.gui.done:
            # update the map
            # print(path)
            # drive gui
            self.gui.run_game(path=path)
            nr_step+=1
            self.new_position = self.gui.current
            new_observation = self.gui.observation
            new_map = self.gui.world

            """
            if new_observation is not None:
                if new_observation["type"] == OBSTACLE:
                    dstar.global_map.set_obstacle(pos=new_observation["pos"])
                if new_observation["pos"] == UNOCCUPIED:
                    dstar.global_map.remove_obstacle(pos=new_observation["pos"])
            """

            if new_observation is not None:
                old_map = new_map
                self.slam.set_ground_truth_map(gt_map=new_map)

            if self.new_position != self.last_position:
                self.last_position = self.new_position

                # slam
                new_edges_and_old_costs, slam_map = self.slam.rescan(global_position=self.new_position)

                self.dstar.new_edges_and_old_costs = new_edges_and_old_costs
                self.dstar.sensed_map = slam_map

                # d star
                path, g, rhs = self.dstar.move_and_replan(robot_position=self.new_position)
        return nr_step