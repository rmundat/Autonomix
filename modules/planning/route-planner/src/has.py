import numpy as np
from time import time
import heapq
import math
import collections
import logging
from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.append('C:/Apps/sourceDocker/code-challenge/Python')
from ML.utils import normalize_angle, LineSegment2d, Vec2d, Box2d
from ML.config import vehicle_param, planner_open_space_config
from ML.grid import Node3d, GridSearch
from ML.rsp import ReedSheppPath, ReedShepp

obstacles_linesegments_vec: List[List[LineSegment2d]] = []

# Define the start and end positions and orientations
sx, sy, sphi = -15.0, -5.0, 0.0
ex, ey, ephi = 15.0, 10.0, 0.0

# Define multiple obstacles, each as a list of Vec2d points (x, y)
obstacles_vertices_vec = [
    # [Vec2d(1.0, 0.0), Vec2d(-1.0, 0.0)],
    # [Vec2d(5.0, 5.0), Vec2d(7.0, 5.0), Vec2d(6.0, 8.0)],
    [Vec2d(-5.0, 2.0), Vec2d(-7.0, 2.0), Vec2d(-6.0, -4.0), Vec2d(-4.0, -3.0)]
]

# Define the boundary of the area
XYbounds = [-50.0, 50.0, -50.0, 50.0]
        
class Clock:
    @staticmethod
    def NowInSeconds():
        return time()
    
class HybridAStartResult:
    def __init__(self):
        self.x = []
        self.y = []
        self.phi = []
        self.v = []
        self.a = []
        self.steer = []
        self.accumulated_s = []  

class HybridAStar:
    def __init__(self, planner_open_space_config, vehicle_param, XYbounds, obstacles_linesegments_vec):
        self.XYbounds_ = XYbounds
        self.planner_open_space_config_ = planner_open_space_config
        self.vehicle_param_ = vehicle_param
        self.reed_shepp_generator = ReedShepp()
        
        self.step_size = planner_open_space_config["warm_start_config"]["step_size"]
        self.traj_forward_penalty = planner_open_space_config["warm_start_config"]["traj_forward_penalty"]
        self.traj_back_penalty = planner_open_space_config["warm_start_config"]["traj_back_penalty"]
        self.traj_gear_switch_penalty = planner_open_space_config["warm_start_config"]["traj_gear_switch_penalty"]
        self.traj_steer_penalty = planner_open_space_config["warm_start_config"]["traj_steer_penalty"]
        self.traj_steer_change_penalty = planner_open_space_config["warm_start_config"]["traj_steer_change_penalty"]
        self.next_node_num = planner_open_space_config["warm_start_config"]["next_node_num"]
        self.astar_max_search_time = planner_open_space_config["warm_start_config"]["astar_max_search_time"]
        self.max_steer_angle = vehicle_param["max_steer_angle"] / vehicle_param["steer_ratio"] * \
            planner_open_space_config["warm_start_config"]["traj_kappa_contraint_ratio"]
        self.xy_grid_resolution = planner_open_space_config['warm_start_config']['grid_a_star_xy_resolution']
        self.arc_length = planner_open_space_config['warm_start_config']["phi_grid_resolution"] * \
            vehicle_param["wheel_base"] / math.tan(self.max_steer_angle * 2 / (self.next_node_num / 2 - 1))

        if self.arc_length < math.sqrt(2) * self.xy_grid_resolution:
            self.arc_length = math.sqrt(2) * self.xy_grid_resolution
        
        self.dp_map_ = {}
        self.open_pq_ = []  
        self.open_set_ = set()
        self.close_set_ = set()  
        self.start_node_ = None
        self.end_node_ = None
        
        self.grid_a_star_heuristic_generator = GridSearch(planner_open_space_config, XYbounds)
        self.obstacles_linesegments_vec_ = obstacles_linesegments_vec

    def CalculateNodeCost(self, current_node, next_node):
        next_node.SetTrajCost(current_node.GetTrajCost() + self.TrajCost(current_node, next_node))
        # evaluate heuristic cost
        optimal_path_cost = 0.0
        optimal_path_cost += self.HoloObstacleHeuristic(next_node)
        next_node.SetHeuCost(optimal_path_cost)

    def TrajCost(self, current_node, next_node):
        # evaluate cost on the trajectory and add current cost
        piecewise_cost = 0.0
        if next_node.GetDirec():
            piecewise_cost += (next_node.GetStepSize() - 1) * self.step_size * self.traj_forward_penalty
        else:
            piecewise_cost += (next_node.GetStepSize() - 1) * self.step_size * self.traj_back_penalty

        if current_node.GetDirec() != next_node.GetDirec():
            piecewise_cost += self.traj_gear_switch_penalty

        piecewise_cost += self.traj_steer_penalty * abs(next_node.GetSteer())
        piecewise_cost += self.traj_steer_change_penalty * abs(next_node.GetSteer() - current_node.GetSteer())
        return piecewise_cost

    def HoloObstacleHeuristic(self, next_node):
        return self.grid_a_star_heuristic_generator.check_dp_map(next_node.GetX(), next_node.GetY(), self.dp_map_)

    def AddNodeToOpenSet(self, current_node, next_node):
        if next_node.GetIndex() not in self.open_set:
            start_time = Clock.NowInSeconds()
            self.CalculateNodeCost(current_node, next_node)
            end_time = Clock.NowInSeconds()
            self.heuristic_time += end_time - start_time
            self.open_set.add(next_node.GetIndex())
            self.open_pq[next_node] = next_node.GetCost()
    
    def NextNodeGenerator(self, current_node, next_node_index):
        if next_node_index < self.next_node_num / 2:
            steering = -self.max_steer_angle + (2 * self.max_steer_angle / (self.next_node_num / 2 - 1)) * next_node_index
            traveled_distance = self.step_size
        else:
            index = next_node_index - self.next_node_num / 2
            steering = -self.max_steer_angle + (2 * self.max_steer_angle / (self.next_node_num / 2 - 1)) * index
            traveled_distance = -self.step_size

        intermediate_x, intermediate_y, intermediate_phi = [current_node.x], [current_node.y], [current_node.phi]

        for i in range(int(math.ceil(self.arc_length / self.step_size))):
            next_phi = intermediate_phi[-1] + traveled_distance / vehicle_param["wheel_base"] * math.tan(steering)
            next_x = intermediate_x[-1] + traveled_distance * math.cos((intermediate_phi[-1] + next_phi) / 2)
            next_y = intermediate_y[-1] + traveled_distance * math.sin((intermediate_phi[-1] + next_phi) / 2)
            
            intermediate_x.append(next_x)
            intermediate_y.append(next_y)
            intermediate_phi.append(normalize_angle(next_phi))

        if (intermediate_x[-1] > XYbounds[1] or intermediate_x[-1] < XYbounds[0] or
                intermediate_y[-1] > XYbounds[3] or intermediate_y[-1] < XYbounds[2]):
            return None

        next_node = Node3d(intermediate_x, intermediate_y, intermediate_phi, XYbounds)
        next_node.SetPre(current_node)
        next_node.SetDirec(traveled_distance > 0)
        next_node.SetSteer(steering)

        return next_node

    def AnalyticExpansion(self, current_node, candidate_final_node):
        reeds_shepp_to_check = ReedSheppPath()
        if not self.reed_shepp_generator.shortest_rsp(current_node, self.end_node_, reeds_shepp_to_check):
            return False
        if not self.RSPCheck(reeds_shepp_to_check):
            return False
        candidate_final_node = self.LoadRSPinCS(reeds_shepp_to_check, current_node)
        return True, candidate_final_node
    
    def RSPCheck(self, reeds_shepp_to_end):
        node = Node3d(reeds_shepp_to_end.x, reeds_shepp_to_end.y, reeds_shepp_to_end.phi, self.XYbounds_)
        return self.ValidityCheck(node)
    
    def LoadRSPinCS(self, reeds_shepp_to_end, current_node):
        # The equivalent of 'new Node3d(...)' in Python would be simply 'Node3d(...)'
        end_node = Node3d(reeds_shepp_to_end.x, reeds_shepp_to_end.y, reeds_shepp_to_end.phi, XYbounds)
        end_node.SetPre(current_node)
        end_node.SetTrajCost(current_node.GetTrajCost() + reeds_shepp_to_end.cost)
        return end_node
    
    def ValidityCheck(self, node):
        if node is None:
            raise ValueError("Node should not be None")
        if node.GetStepSize() <= 0:
            raise ValueError("Node step size should be greater than 0")

        if not self.obstacles_linesegments_vec_:
            return True

        node_step_size = node.GetStepSize()
        traversed_x = node.GetXs()
        traversed_y = node.GetYs()
        traversed_phi = node.GetPhis()
        
        if node_step_size > 160:
            print(f"node step size is: {node_step_size}")

        # The first {x, y, phi} is collision free unless they are start and end
        # configuration of search problem
        check_start_index = 0 if node_step_size == 1 else 1

        for i in range(check_start_index, node_step_size):
            if (traversed_x[i] > self.XYbounds_[1] or traversed_x[i] < self.XYbounds_[0]
                or traversed_y[i] > self.XYbounds_[3] or traversed_y[i] < self.XYbounds_[2]):
                return False

            bounding_box = Node3d.GetBoundingBox(self.vehicle_param_, traversed_x[i], traversed_y[i], traversed_phi[i])

            for obstacle_linesegments in self.obstacles_linesegments_vec_:
                for linesegment in obstacle_linesegments:
                    if bounding_box.has_overlap(linesegment):
                        print(f"collision start at x: {linesegment.start.x}")
                        print(f"collision start at y: {linesegment.start.y}")
                        print(f"collision end at x: {linesegment.end.x}")
                        print(f"collision end at y: {linesegment.end.y}")
                        return False
        return True
    
    
    def Plan(self, sx, sy, sphi, ex, ey, ephi, XYbounds, obstacles_vertices_vec, result):
        # clear containers
        self.open_set_.clear()
        self.close_set_.clear()
        self.final_node_ = None

        # Convert obstacle vertices into line segments and plot them        
        for obstacle_vertices in obstacles_vertices_vec:
            vertices_num = len(obstacle_vertices)
            obstacle_linesegments: List[LineSegment2d] = []
            for i in range(vertices_num - 1):
                line_segment = LineSegment2d(obstacle_vertices[i], obstacle_vertices[i + 1])
                obstacle_linesegments.append(line_segment)
            obstacles_linesegments_vec.append(obstacle_linesegments)
            
        obstacles_linesegments_vec_ = obstacles_linesegments_vec
        for i in range(len(obstacles_linesegments_vec_)):
            for linesg in obstacles_linesegments_vec_[i]:
                name = str(i) + "roi_boundary"

        sposition = Vec2d(sx, sy)
        svec_to_center = Vec2d((vehicle_param['front_edge_to_center'] - vehicle_param['back_edge_to_center']) / 2.0,
                               (vehicle_param['left_edge_to_center'] - vehicle_param['right_edge_to_center']) / 2.0)
        scenter = sposition + svec_to_center.rotate(sphi)
        sbox = Box2d(scenter, sphi, vehicle_param['length'], vehicle_param['width'])
        eposition = Vec2d(ex, ey)
        evec_to_center = Vec2d((vehicle_param['front_edge_to_center'] - vehicle_param['back_edge_to_center']) / 2.0,
                               (vehicle_param['left_edge_to_center'] - vehicle_param['right_edge_to_center']) / 2.0)
        ecenter = eposition + evec_to_center.rotate(ephi)
        ebox = Box2d(ecenter, ephi, vehicle_param['length'], vehicle_param['width'])
        self.XYbounds_ = XYbounds
        
        # load nodes and obstacles
        self.start_node_ = Node3d([sx], [sy], [sphi], XYbounds)
        self.end_node_ = Node3d([ex], [ey], [ephi], XYbounds)
        print("start node", sx, sy, sphi)
        print("end node", ex, ey, ephi)
        
        if not self.ValidityCheck(self.start_node_):
            logging.error("start_node in collision with obstacles")
            logging.error(f"{self.start_node_.GetX()},{self.start_node_.GetY()},{self.start_node_.GetPhi()}")
            return False
        if not self.ValidityCheck(self.end_node_):
            logging.error("end_node in collision with obstacles")
            return False
        
        map_time = Clock.NowInSeconds()
        grid_search = GridSearch(planner_open_space_config, XYbounds)
        success = grid_search.generate_dp_map(ex, ey, XYbounds, obstacles_linesegments_vec, self.open_pq_, self.dp_map_)
        print("map time", Clock.NowInSeconds() - map_time)
        
        # load open set, pq
        self.open_set_.add(self.start_node_.GetIndex())
        heapq.heappush(self.open_pq_, (self.start_node_, self.start_node_.GetCost()))

        # Hybrid A* begins
        explored_node_num = 0
        available_result_num = 0
        best_explored_num = explored_node_num
        best_available_result_num = available_result_num
        astar_start_time = Clock.NowInSeconds()
        heuristic_time = 0.0
        rs_time = 0.0
        node_generator_time = 0.0
        validity_check_time = 0.0
        max_explored_num = planner_open_space_config['warm_start_config']['max_explored_num']
        desired_explored_num = min(planner_open_space_config['warm_start_config']['desired_explored_num'],
                                planner_open_space_config['warm_start_config']['max_explored_num'])
        kMaxNodeNum = 200000
        candidate_final_nodes = []
        
        while self.open_pq_ and len(self.open_pq_) < kMaxNodeNum and available_result_num < desired_explored_num and explored_node_num < max_explored_num:
            current_node = heapq.heappop(self.open_pq_)[0]
            rs_start_time = Clock.NowInSeconds()
            final_node = None
            success, final_node = self.AnalyticExpansion(current_node, final_node)
            if success:
                if self.final_node_ is None or self.final_node_.GetTrajCost() > final_node.GetTrajCost():
                    self.final_node_ = final_node
                    best_explored_num = explored_node_num + 1
                    best_available_result_num = available_result_num + 1
                available_result_num += 1
            explored_node_num += 1
            rs_end_time = Clock.NowInSeconds()
            rs_time += rs_end_time - rs_start_time
            self.close_set_.add(current_node.GetIndex())

            if Clock.NowInSeconds() - astar_start_time > self.astar_max_search_time and available_result_num > 0:
                break
            else:
                print("Hybrid A* search time", Clock.NowInSeconds() - astar_start_time)
            
            begin_index = 0
            end_index = self.next_node_num
            temp_set = set()
            for i in range(begin_index, end_index - 1):
                gen_node_time = Clock.NowInSeconds()
                next_node = self.NextNodeGenerator(current_node, i)
                node_generator_time += Clock.NowInSeconds() - gen_node_time

                # boundary check failure handle
                if next_node is None:
                    continue
                # check if the node is already in the close set
                if next_node.GetIndex() in self.close_set_:
                    continue
                # collision check
                validity_check_start_time = Clock.NowInSeconds()
                if not self.ValidityCheck(next_node):
                    continue
                validity_check_time += Clock.NowInSeconds() - validity_check_start_time
                if next_node.GetIndex() not in self.open_set_:
                    start_time = Clock.NowInSeconds()
                    self.CalculateNodeCost(current_node, next_node)
                    end_time = Clock.NowInSeconds()
                    heuristic_time += end_time - start_time
                    temp_set.add(next_node.GetIndex())
                    heapq.heappush(self.open_pq_, (next_node, next_node.GetCost()))
            self.open_set_.update(temp_set)

        if self.final_node_ is None:
            print("Hybrid A* cannot find a valid path")
            return False
        
        print(f"open_pq_.empty(): {'true' if not self.open_pq_ else 'false'}")
        print(f"open_pq_.size(): {len(self.open_pq_)}")
        print(f"desired_explored_num: {desired_explored_num}")
        print(f"min cost is: {self.final_node_.GetTrajCost()}")
        print(f"max_explored_num is: {max_explored_num}")
        print(f"explored node num is: {explored_node_num}, available_result_num: {available_result_num}")
        print(f"best_explored_num is: {best_explored_num}, best_available_result_num is: {best_available_result_num}")
        print(f"cal node time is: {heuristic_time}, validity_check_time: {validity_check_time}, node_generator_time: {node_generator_time}")
        print(f"reed shepp time is: {rs_time}")
        print(f"hybrid astar total time is: {Clock.NowInSeconds() - astar_start_time}")

        return True

# Function to create a rectangle representing the vehicle at a given position and orientation
def create_vehicle_box(center_x, center_y, phi, length, width):
    # Calculate the four corners of the vehicle box, assuming center is at the origin
    # and then rotate and translate the points according to the vehicle's position and orientation
    corner_offsets = [(length / 2.0, width / 2.0), (-length / 2.0, width / 2.0),
        (-length / 2.0, -width / 2.0), (length / 2.0, -width / 2.0)]
    corners = []
    for x_off, y_off in corner_offsets:
        # Rotate the corners around the origin and then translate
        rotated_x = x_off * np.cos(phi) - y_off * np.sin(phi)
        rotated_y = x_off * np.sin(phi) + y_off * np.cos(phi)
        corners.append((center_x + rotated_x, center_y + rotated_y))
    return corners

def plot_map(dp_map, xy_bounds, hybrid_astar, obstacles_vertices_vec):
    # Create a 2D grid based on the xy_bounds and xy_resolution
    xy_grid_resolution = planner_open_space_config['warm_start_config']['grid_a_star_xy_resolution']
    x_min, x_max, y_min, y_max = xy_bounds
    x_range = np.arange(x_min, x_max + xy_grid_resolution, xy_grid_resolution)
    y_range = np.arange(y_min, y_max + xy_grid_resolution, xy_grid_resolution)
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    # Create a 2D array to store the cost values
    cost_map = np.zeros_like(grid_x)

    # Fill the cost_map with the cost values from the dp_map
    for index, node in dp_map.items():
        grid_x_index = node.get_grid_x()
        grid_y_index = node.get_grid_y()
        cost_map[grid_y_index, grid_x_index] = node.get_cost()

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the cost map as a heatmap
    heatmap = ax.pcolormesh(grid_x, grid_y, cost_map, cmap='viridis')

    # Add a colorbar to show the cost values
    cbar = fig.colorbar(heatmap)
    cbar.set_label('Cost to Goal')

    # Extract the path from the final node
    path = []
    node = hybrid_astar.final_node_
    while node is not None:
        path.append((node.GetX(), node.GetY()))
        node = node.GetPreNode()
    path.reverse()

    # Plot the path
    path_x, path_y = zip(*path)
    ax.plot(path_x, path_y, '-b', linewidth=2, label='Hybrid A* Path')

    # Plot the start and goal positions
    ax.plot(hybrid_astar.start_node_.GetX(), hybrid_astar.start_node_.GetY(), 'ro', markersize=10, label='Start')
    ax.plot(hybrid_astar.end_node_.GetX(), hybrid_astar.end_node_.GetY(), 'go', markersize=10, label='Goal')

    # Plot the obstacles with larger boundaries and a different color
    for i, obstacle_vertices in enumerate(obstacles_vertices_vec):
        obstacle_polygon = patches.Polygon([(v.x(), v.y()) for v in obstacle_vertices], closed=True,
                                           facecolor='cyan', edgecolor='red', linewidth=2, alpha=0.7)
        ax.add_patch(obstacle_polygon)
        
        # Calculate the centroid of the obstacle polygon
        # centroid = np.mean([(v.x(), v.y()) for v in obstacle_vertices], axis=0)
        
        # Add a text label at the centroid of the obstacle
        # ax.text(centroid[0], centroid[1], f'Obstacle {i+1}', fontsize=12, ha='center', va='center', color='black')

    # Set the axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('DP Map with Hybrid A* Path and Obstacles')
    ax.legend()

    # Show the plot
    plt.show()
    plt.show()
    
    
# Instance of the HybridAStar class
hybrid_astar = HybridAStar(planner_open_space_config, vehicle_param, XYbounds, obstacles_linesegments_vec)

result = HybridAStartResult()
success = hybrid_astar.Plan(sx, sy, sphi, ex, ey, ephi, XYbounds, obstacles_vertices_vec, result)

# Plot the dp_map
if success:
    plot_map(hybrid_astar.dp_map_, XYbounds, hybrid_astar, obstacles_vertices_vec)
else:
    print("Failed to find a valid path")