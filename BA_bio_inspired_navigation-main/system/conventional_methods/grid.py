import numpy as np
from system.conventional_methods.utils import get_movements_4n, get_movements_8n, heuristic, Vertices, Vertex
from typing import Dict, List

OBSTACLE = 255
UNOCCUPIED = 0

def get_environment3():
    grid = np.zeros((10,10), dtype=np.uint8)
    for x in range(5,8):
        for xx in range(5,8):
            grid[x][xx]=OBSTACLE
    start = (0,0)
    goal = (9,9)
    return grid, start, goal

def get_environment2():
    grid = np.zeros((40,40), dtype=np.uint8)
    for x in range(20,30):
        for xx in range(20,30):
            grid[39-x][xx]=OBSTACLE
    start = (39-0,0)
    goal = (39-35,35)
    return grid, start, goal

def get_environment(env="plane_doors", scale = 1):
    grid = np.zeros((int(110*scale), int(110*scale)), dtype=np.uint8)
    dic = {
        "plane_doors"  : [1, 3, 5, 7],
        "plane_doors_1" : [3, 5, 7],
        "plane_doors_2" : [1, 5, 7],
        "plane_doors_3" : [1, 3, 7],
        "plane_doors_4" : [1, 3, 5],
        "plane_doors_5c_3o" : [1, 3, 7, 9],
        "plane_doors_5c_2o" : [1,5,7,9],
        "plane" : [ ],
        "plane_unstructured_doors": [1, 3, 5, 7],
        "plane_unstructured" : [],
        "plane_undiscovered_blank": [],
        "plane_undiscovered_block": []
    }
    doors = dic[env]#[1, 3, 5, 7]# [1, 3, 5, 7]
    if env == "plane_unstructured" or env == "plane_unstructured_doors":
        unstructured=True
        triangel_y = [4.0, 4.5, 4.0, 3.0, 4.0, 5.0]
        box_y = [8, 8.5, 7.5, 7.5, 8.5]
        triangel_y = [4.0, 4.5, 4.0, 3.0, 4.0, 5.0]
        box_y = [8, 8.5, 7.5, 7.5, 8.5]
        door_y = [4.4,4.4,4.4,4.4,4.4]
    else:
        triangel_y = [5, 5, 5, 5, 5, 5]
        box_y = [8, 8, 8, 8, 8]
        door_y = [5.4,5.4,5.4,5.4,5.4]




    for id, x in enumerate(doors):
        base_coor = [int(x*10*scale),int(door_y[id]*10*scale)]
        rec = [int(10*scale),int(2*scale)]
        for i in range(base_coor[0],base_coor[0]+rec[0]):
            for ii in range(base_coor[1], base_coor[1] + rec[1]):
                grid[ii][i] = OBSTACLE#grid[109-ii][i] = OBSTACLE
    boxes = [0, 2, 4, 6, 8, 10]
    if env == "plane_undiscovered_blank":  # the area of the first door is set as blank
        boxes = [-1,2,4,6,8,10] # -1 is a place holder
    elif env == "plane_undiscovered_block":  # the first door of  the first door is set as all obstacles
        boxes = [-1,2,4,6,8,10]
        #the undiscovered area is set as obstacles
        for i in range(0,(10+10+5)*scale):
            for ii in range(0, (50+20)*scale):
                grid[ii][i] = OBSTACLE
    else:
        boxes=boxes

    for id, x in enumerate(boxes):
        if x == -1:
            continue
        base_coor = [int(x*10*scale),int(triangel_y[id]*10*scale)]
        rec = [int(10*scale),int(20*scale)]
        for i in range(base_coor[0],base_coor[0]+rec[0]):
            for ii in range(base_coor[1], base_coor[1]+rec[1]):
                grid[ii][i] = OBSTACLE#grid[109 - ii][i] = OBSTACLE


    boxes = [1, 3, 5, 7, 9]
    for id, x in enumerate(boxes):
        base_coor = [int(x*10*scale), int(box_y[id]*10*scale)]
        rec = [int(10*scale), int(10*scale)]
        for i in range(base_coor[0],base_coor[0]+rec[0]+1):
            for ii in range(base_coor[1], base_coor[1] + rec[1] + 1):
                grid[ii][i] = OBSTACLE#grid[109 - ii][i] = OBSTACLE

    start = (int(6*scale),int(55*scale))#(109 - 6,55)
    goal = (int(100*scale),int(15*scale))#(109 - 100,15)


    return grid, start, goal



class OccupancyGridMap:
    def __init__(self, x_dim, y_dim, exploration_setting='4N', own_map = None):
        """
        set initial values for the map occupancy grid
        |----------> y, column
        |           (x=0,y=2)
        |
        V (x=2, y=0)
        x, row
        :param x_dim: dimension in the x direction
        :param y_dim: dimension in the y direction
        """


        if own_map is None:
            self.x_dim = x_dim
            self.y_dim = y_dim

            # the map extents in units [m]
            self.map_extents = (x_dim, y_dim)
            # the obstacle map
            self.occupancy_grid_map = np.zeros(self.map_extents, dtype=np.uint8)
        else:
            self.x_dim = own_map.shape[0]
            self.y_dim = own_map.shape[1]
            self.map_extents = own_map.shape
            self.occupancy_grid_map = np.copy(own_map)


        # obstacles
        self.visited = {}
        self.exploration_setting = exploration_setting

    def get_map(self):
        """
        :return: return the current occupancy grid map
        """
        return self.occupancy_grid_map

    def set_map(self, new_ogrid):
        """
        :param new_ogrid:
        :return: None
        """
        self.occupancy_grid_map = new_ogrid

    def is_unoccupied(self, pos: (int, int)) -> bool:
        """
        :param pos: cell position we wish to check
        :return: True if cell is occupied with obstacle, False else
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)

        # if not self.in_bounds(cell=(x, y)):
        #    raise IndexError("Map index out of bounds")

        return self.occupancy_grid_map[row][col] == UNOCCUPIED

    def in_bounds(self, cell: (int, int)) -> bool:
        """
        Checks if the provided coordinates are within
        the bounds of the grid map
        :param cell: cell position (x,y)
        :return: True if within bounds, False else
        """
        (x, y) = cell
        return 0 <= x < self.x_dim and 0 <= y < self.y_dim

    def filter(self, neighbors: List, avoid_obstacles: bool):
        """
        :param neighbors: list of potential neighbors before filtering
        :param avoid_obstacles: if True, filter out obstacle cells in the list
        :return:
        """
        if avoid_obstacles:
            return [node for node in neighbors if self.in_bounds(node) and self.is_unoccupied(node)]
        return [node for node in neighbors if self.in_bounds(node)]

    def succ(self, vertex: (int, int), avoid_obstacles: bool = False) -> list:
        """
        :param avoid_obstacles:
        :param vertex: vertex you want to find direct successors from
        :return:
        """
        (x, y) = vertex

        if self.exploration_setting == '4N':  # change this
            movements = get_movements_4n(x=x, y=y)
        else:
            movements = get_movements_8n(x=x, y=y)

        # not needed. Just makes aesthetics to the path
        if (x + y) % 2 == 0: movements.reverse()

        filtered_movements = self.filter(neighbors=movements, avoid_obstacles=avoid_obstacles)
        return list(filtered_movements)

    def set_obstacle(self, pos: (int, int)):
        """
        :param pos: cell position we wish to set obstacle
        :return: None
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)
        self.occupancy_grid_map[row, col] = OBSTACLE

    def remove_obstacle(self, pos: (int, int)):
        """
        :param pos: position of obstacle
        :return: None
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)
        self.occupancy_grid_map[row, col] = UNOCCUPIED

    def local_observation(self, global_position: (int, int), view_range: int = 2) -> Dict:
        """
        :param global_position: position of robot in the global map frame
        :param view_range: how far ahead we should look
        :return: dictionary of new observations
        """
        (px, py) = global_position
        nodes = [(x, y) for x in range(px - view_range, px + view_range + 1)
                 for y in range(py - view_range, py + view_range + 1)
                 if self.in_bounds((x, y))]
        return {node: UNOCCUPIED if self.is_unoccupied(pos=node) else OBSTACLE for node in nodes}


class SLAM:
    def __init__(self, map: OccupancyGridMap, view_range: int):
        self.ground_truth_map = map
        self.slam_map = OccupancyGridMap(x_dim=map.x_dim,
                                         y_dim=map.y_dim,
                                         )
        self.view_range = view_range

    def set_ground_truth_map(self, gt_map: OccupancyGridMap):
        self.ground_truth_map = gt_map

    def c(self, u: (int, int), v: (int, int)) -> float:
        """
        calcuclate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance to traverse. inf if obstacle in path
        """
        if not self.slam_map.is_unoccupied(u) or not self.slam_map.is_unoccupied(v):
            return float('inf')
        else:
            return heuristic(u, v)

    def rescan(self, global_position: (int, int)):

        # rescan local area
        local_observation = self.ground_truth_map.local_observation(global_position=global_position,
                                                                    view_range=self.view_range)

        vertices = self.update_changed_edge_costs(local_grid=local_observation)
        return vertices, self.slam_map

    def update_changed_edge_costs(self, local_grid: Dict) -> Vertices:
        vertices = Vertices()
        for node, value in local_grid.items():
            # if obstacle
            if value == OBSTACLE:
                if self.slam_map.is_unoccupied(node):
                    v = Vertex(pos=node)
                    succ = self.slam_map.succ(node)
                    for u in succ:
                        v.add_edge_with_cost(succ=u, cost=self.c(u, v.pos))
                    vertices.add_vertex(v)
                    self.slam_map.set_obstacle(node)
            else:
                # if white cell
                if not self.slam_map.is_unoccupied(node):
                    v = Vertex(pos=node)
                    succ = self.slam_map.succ(node)
                    for u in succ:
                        v.add_edge_with_cost(succ=u, cost=self.c(u, v.pos))
                    vertices.add_vertex(v)
                    self.slam_map.remove_obstacle(node)
        return vertices
