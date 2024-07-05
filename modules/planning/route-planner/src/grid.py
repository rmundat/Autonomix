import numpy as np
import math
import heapq
import matplotlib.patches as patches
from typing import List, Dict, Tuple
from ML.config import planner_open_space_config
from ML.utils import LineSegment2d, Vec2d, Box2d

class Node:
    def __init__(self, index, cost):
        self.index = index
        self.cost = cost

    def __lt__(self, other):
        return other.cost >= self.cost

class Node2d:
    def __init__(self, x: float, y: float, xy_resolution: float, XYbounds: List[float]):
        self.path_cost_ = 0.0
        self.heuristic_ = 0.0
        self.cost_ = 0.0
        self.pre_node_ = None
        self.grid_x_ = int((x - XYbounds[0]) / xy_resolution)
        self.grid_y_ = int((y - XYbounds[2]) / xy_resolution)
        self.index_ = self.compute_string_index(self.grid_x_, self.grid_y_)
        
    @classmethod
    def node2d(cls, grid_x: int, grid_y: int, XYbounds):
        node = cls.__new__(cls)
        node.path_cost_ = 0.0
        node.heuristic_ = 0.0
        node.cost_ = 0.0
        node.pre_node_ = None
        node.grid_x_ = int(grid_x)
        node.grid_y_ = int(grid_y)
        node.index_ = node.compute_string_index(node.grid_x_, node.grid_y_)
        return node

    @staticmethod
    def compute_string_index(x_grid: int, y_grid: int) -> str:
        return f"{x_grid}_{y_grid}"

    def set_path_cost(self, path_cost: float):
        self.path_cost_ = path_cost
        self.cost_ = self.path_cost_ + self.heuristic_

    def set_heuristic(self, heuristic: float):
        self.heuristic_ = heuristic
        self.cost_ = self.path_cost_ + self.heuristic_

    def set_cost(self, cost: float):
        self.cost_ = cost

    def set_pre_node(self, pre_node):
        self.pre_node_ = pre_node

    def get_grid_x(self) -> int:
        return self.grid_x_

    def get_grid_y(self) -> int:
        return self.grid_y_

    def get_path_cost(self) -> float:
        return self.path_cost_

    def get_heuristic(self) -> float:
        return self.heuristic_

    def get_cost(self) -> float:
        return self.cost_

    def get_index(self) -> str:
        return self.index_

    def get_pre_node(self):
        return self.pre_node_

    @staticmethod
    def calc_index(x: float, y: float, xy_resolution: float, XYbounds: List[float]) -> str:
        grid_x = int((x - XYbounds[0]) / xy_resolution)
        grid_y = int((y - XYbounds[2]) / xy_resolution)
        return Node2d.compute_string_index(grid_x, grid_y)

    def __eq__(self, other):
        return self.get_index() == other.get_index()
    
class Node3d:
    def __init__(self, traversed_x, traversed_y, traversed_phi, XYbounds):
        
        self.traj_cost = 0.0
        self.heuristic_cost = 0.0
        self.cost = 0.0
        self.pre_node = None
        self.steering = 0.0
        self.direction = True  # true for moving forward and false for moving backward
        
        self.x = traversed_x[-1]
        self.y = traversed_y[-1]
        self.phi = traversed_phi[-1]
        
        # Assuming xy_grid_resolution and phi_grid_resolution are known and defined
        self.xy_grid_resolution_ = planner_open_space_config['warm_start_config']['xy_grid_resolution']
        self.phi_grid_resolution_ = planner_open_space_config['warm_start_config']['phi_grid_resolution']

        self.x_grid = int((self.x - XYbounds[0]) / self.xy_grid_resolution_)
        self.y_grid = int((self.y - XYbounds[2]) / self.xy_grid_resolution_)
        self.phi_grid = int((self.phi - (-np.pi)) / self.phi_grid_resolution_)
        
        self.traversed_x = [] if traversed_x is None else traversed_x
        self.traversed_y = [] if traversed_y is None else traversed_y
        self.traversed_phi = [] if traversed_phi is None else traversed_phi
        
        self.index = self.ComputeStringIndex(self.x_grid, self.y_grid, self.phi_grid)
        self.step_size = len(self.traversed_x)

    def GetCost(self):
        return self.traj_cost + self.heuristic_cost

    def GetTrajCost(self):
        return self.traj_cost

    def GetHeuCost(self):
        return self.heuristic_cost

    def GetGridX(self):
        return self.x_grid

    def GetGridY(self):
        return self.y_grid

    def GetX(self):
        return self.x

    def GetY(self):
        return self.y

    def GetPhi(self):
        return self.phi

    def GetIndex(self):
        return self.index

    def GetStepSize(self):
        return self.step_size

    def GetDirec(self):
        return self.direction

    def GetSteer(self):
        return self.steering

    def GetPreNode(self):
        return self.pre_node

    def GetXs(self):
        return self.traversed_x

    def GetYs(self):
        return self.traversed_y

    def GetPhis(self):
        return self.traversed_phi

    def SetPre(self, pre_node):
        self.pre_node = pre_node

    def SetDirec(self, direction):
        self.direction = direction

    def SetTrajCost(self, cost):
        self.traj_cost = cost

    def SetHeuCost(self, cost):
        self.heuristic_cost = cost

    def SetSteer(self, steering):
        self.steering = steering
        
    # This is where you need to adjust the coordinates for the bounding box
    @staticmethod
    def GetBoundingBox(vehicle_param, x, y, phi):
        ego_length = vehicle_param["length"]
        ego_width = vehicle_param["width"]
        shift_distance = ego_length / 2.0 - vehicle_param["back_edge_to_center"]
        
        center_x = x + shift_distance * math.cos(phi)
        center_y = y + shift_distance * math.sin(phi)
        center = Vec2d(center_x, center_y)
        
        ego_box = Box2d(center, ego_length, ego_width, phi)
        return ego_box

    @staticmethod
    def ComputeStringIndex(x_grid, y_grid, phi_grid):
        return f"{x_grid}_{y_grid}_{phi_grid}"
    
    def __eq__(self, other):
        return self.GetIndex() == other.GetIndex()
    
    def __lt__(self, other):
        return other.cost >= self.cost

class GridAStarResult:
    def __init__(self):
        self.x = []
        self.y = []
        self.path_cost = 0.0

class GridSearch:
    def __init__(self, open_space_conf, XYbounds):
        self.xy_grid_resolution_ = open_space_conf["warm_start_config"]["grid_a_star_xy_resolution"]
        self.node_radius_ = open_space_conf["warm_start_config"]["node_radius"]
        self.XYbounds_ = XYbounds
        self.max_grid_x_ = 0.0
        self.max_grid_y_ = 0.0
        self.start_node_ = None
        self.end_node_ = None
        self.final_node_ = None
        self.obstacles_linesegments_vec_ = []
        # self.dp_map_ = {}

    def euclidean_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def check_constraints(self, node: Node2d) -> bool:
        node_grid_x = node.get_grid_x()
        node_grid_y = node.get_grid_y()
        if (node_grid_x > self.max_grid_x_ or node_grid_x < 0 or node_grid_y > self.max_grid_y_ or node_grid_y < 0):
            return False
        if not self.obstacles_linesegments_vec_:
            return True
        for obstacle_linesegments in self.obstacles_linesegments_vec_:
            for linesegment in obstacle_linesegments:
                if linesegment.distance_to(Vec2d(node.get_grid_x(), node.get_grid_y())) < self.node_radius_:
                    return False
        return True

    def generate_next_nodes(self, current_node: Node2d, XYbounds) -> List[Node2d]:
        current_node_x = current_node.get_grid_x()
        current_node_y = current_node.get_grid_y()
        current_node_path_cost = current_node.get_path_cost()
        diagonal_distance = math.sqrt(2.0)
        self.XYbounds_ = XYbounds
        next_nodes = []

        up = Node2d.node2d(current_node_x, current_node_y + 1.0, self.XYbounds_)
        up.set_path_cost(current_node_path_cost + 1.0)

        up_right = Node2d.node2d(current_node_x + 1.0, current_node_y + 1.0, self.XYbounds_)
        up_right.set_path_cost(current_node_path_cost + diagonal_distance)

        right = Node2d.node2d(current_node_x + 1.0, current_node_y, self.XYbounds_)
        right.set_path_cost(current_node_path_cost + 1.0)

        down_right = Node2d.node2d(current_node_x + 1.0, current_node_y - 1.0, self.XYbounds_)
        down_right.set_path_cost(current_node_path_cost + diagonal_distance)

        down = Node2d.node2d(current_node_x, current_node_y - 1.0, self.XYbounds_)
        down.set_path_cost(current_node_path_cost + 1.0)

        down_left = Node2d.node2d(current_node_x - 1.0, current_node_y - 1.0, self.XYbounds_)
        down_left.set_path_cost(current_node_path_cost + diagonal_distance)

        left = Node2d.node2d(current_node_x - 1.0, current_node_y, self.XYbounds_)
        left.set_path_cost(current_node_path_cost + 1.0)

        up_left = Node2d.node2d(current_node_x - 1.0, current_node_y + 1.0, self.XYbounds_)
        up_left.set_path_cost(current_node_path_cost + diagonal_distance)

        next_nodes.extend([up, up_right, right, down_right, down, down_left, left, up_left])
        return next_nodes

    def generate_a_star_path(
        self,
        sx: float,
        sy: float,
        ex: float,
        ey: float,
        XYbounds: List[float],
        obstacles_linesegments_vec: List[List[Tuple[float, float]]],
        result: GridAStarResult) -> bool:
        open_pq = []
        open_set = {}
        close_set = {}
        self.XYbounds_ = XYbounds
        start_node = Node2d(sx, sy, self.xy_grid_resolution_, self.XYbounds_)
        end_node = Node2d(ex, ey, self.xy_grid_resolution_, self.XYbounds_)
        self.final_node_ = None
        self.obstacles_linesegments_vec_ = obstacles_linesegments_vec

        open_set[start_node.get_index()] = start_node
        open_pq.append((start_node.get_index(), start_node.get_cost()))

        explored_node_num = 0
        while open_pq:
            current_id, _ = min(open_pq, key=lambda x: x[1])
            open_pq.remove((current_id, _))
            current_node = open_set[current_id]

            if current_node == end_node:
                self.final_node_ = current_node
                break

            close_set[current_node.get_index()] = current_node
            next_nodes = self.generate_next_nodes(current_node)

            for next_node in next_nodes:
                if not self.check_constraints(next_node):
                    continue
                if next_node.get_index() in close_set:
                    continue
                if next_node.get_index() not in open_set:
                    explored_node_num += 1
                    next_node.set_heuristic(
                        self.euclidean_distance(next_node.get_grid_x(), 
                                                next_node.get_grid_y(),
                                                end_node.get_grid_x(), 
                                                end_node.get_grid_y())
                        )
                    next_node.set_pre_node(current_node)
                    open_set[next_node.get_index()] = next_node
                    open_pq.append((next_node.get_index(), next_node.get_cost()))

        if self.final_node_ is None:
            print("Grid A* search returned null ptr (open_set ran out)")
            return False

        self.load_grid_a_star_result(result)
        print(f"Explored node num: {explored_node_num}")
        return True

    def generate_dp_map(self, ex, ey, xy_bounds, obstacles_linesegments_vec, open_pq, dp_map):
        open_set = {}
        self.xy_bounds_ = xy_bounds
        self.max_grid_y_ = int(round((self.xy_bounds_[3] - self.xy_bounds_[2]) / self.xy_grid_resolution_))
        self.max_grid_x_ = int(round((self.xy_bounds_[1] - self.xy_bounds_[0]) / self.xy_grid_resolution_))
        end_node = Node2d(ex, ey, self.xy_grid_resolution_, self.xy_bounds_)
        self.obstacles_linesegments_vec_ = obstacles_linesegments_vec
        open_set[end_node.get_index()] = end_node
        heapq.heappush(open_pq, Node(end_node.get_index(), end_node.get_cost()))
        explored_node_num = 0       
        while open_pq:
            current_id = heapq.heappop(open_pq)
            current_node = open_set[current_id.index]
            dp_map[current_node.get_index()] = current_node
            next_nodes = self.generate_next_nodes(current_node, self.xy_bounds_)
            for next_node in next_nodes:
                if not self.check_constraints(next_node):
                    continue
                if next_node.get_index() in dp_map:
                    continue
                if next_node.get_index() not in open_set:
                    explored_node_num += 1
                    next_node.set_pre_node(current_node)
                    open_set[next_node.get_index()] = next_node
                    heapq.heappush(open_pq, Node(next_node.get_index(), next_node.get_cost()))
                else:
                    if open_set[next_node.get_index()].get_cost() > next_node.get_cost():
                        open_set[next_node.get_index()].set_cost(next_node.get_cost())
                        open_set[next_node.get_index()].set_pre_node(current_node)

        print(f"explored node num is {explored_node_num}")
        return True

    def check_dp_map(self, sx: float, sy: float, dp_map) -> float:
        index = Node2d.calc_index(sx, sy, self.xy_grid_resolution_, self.XYbounds_)
        if index in dp_map:
            return dp_map[index].get_cost() * self.xy_grid_resolution_
        else:
            return float("inf")

    def load_grid_a_star_result(self, result: GridAStarResult):
        result.path_cost = self.final_node_.get_path_cost() * self.xy_grid_resolution_
        current_node = self.final_node_
        grid_a_x = []
        grid_a_y = []
        while current_node.get_pre_node() is not None:
            grid_a_x.append(current_node.get_grid_x() * self.xy_grid_resolution_ + self.XYbounds_[0])
            grid_a_y.append(current_node.get_grid_y() * self.xy_grid_resolution_ + self.XYbounds_[2])
            current_node = current_node.get_pre_node()
        grid_a_x.reverse()
        grid_a_y.reverse()
        result.x = grid_a_x
        result.y = grid_a_y