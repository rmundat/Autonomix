import numpy as np
from time import time
import heapq
import math
import collections
import logging
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.append('C:/Apps/sourceDocker/code-challenge/Python')
from ML.utils import normalize_angle, LineSegment2d, Vec2d, Box2d
from ML.config import vehicle_param, planner_open_space_config
from ML.grid import Node3d, GridSearch
from ML.rsp import ReedSheppPath

# Define the start and end positions and orientations
sx, sy, sphi = -15.0, 0.0, 0.0
ex, ey, ephi = 15.0, 0.0, 0.0

# Define multiple obstacles, each as a list of Vec2d points (x, y)
obstacles_vertices_vec = [
    [Vec2d(1.0, 0.0), Vec2d(-1.0, 0.0)],
    # [Vec2d(5.0, 5.0), Vec2d(7.0, 5.0), Vec2d(6.0, 8.0)],
    # [Vec2d(-5.0, -1.0), Vec2d(-7.0, -1.0), Vec2d(-6.0, -4.0), Vec2d(-4.0, -3.0)]
]

# Define the boundary of the area
XYbounds = [-50.0, 50.0, -50.0, 50.0]

open_set_ = set()
open_pq_ = []
# open_pq = collections.defaultdict(lambda: float('inf'))
        
class Clock:
    @staticmethod
    def NowInSeconds():
        return time.time()  

class HybridAStar:
    def __init__(self, planner_open_space_config, vehicle_param, XYbounds, obstacles_linesegments_vec):
        self.step_size = planner_open_space_config["warm_start_config"]["step_size"]
        self.traj_forward_penalty = planner_open_space_config["warm_start_config"]["traj_forward_penalty"]
        self.traj_back_penalty = planner_open_space_config["warm_start_config"]["traj_back_penalty"]
        self.traj_gear_switch_penalty = planner_open_space_config["warm_start_config"]["traj_gear_switch_penalty"]
        self.traj_steer_penalty = planner_open_space_config["warm_start_config"]["traj_steer_penalty"]
        self.traj_steer_change_penalty = planner_open_space_config["warm_start_config"]["traj_steer_change_penalty"]
        self.grid_a_star_heuristic_generator = GridSearch(planner_open_space_config)
        self.heuristic_time = 0.0
        self.XYbounds_ = XYbounds
        self.planner_open_space_config_ = planner_open_space_config
        self.vehicle_param_ = vehicle_param
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
        return self.grid_a_star_heuristic_generator.CheckDpMap(next_node.GetX(), next_node.GetY())

    def AddNodeToOpenSet(self, current_node, next_node):
        if next_node.GetIndex() not in self.open_set:
            start_time = Clock.NowInSeconds()
            self.CalculateNodeCost(current_node, next_node)
            end_time = Clock.NowInSeconds()
            self.heuristic_time += end_time - start_time
            self.open_set.add(next_node.GetIndex())
            self.open_pq[next_node] = next_node.GetCost()
    
    def NextNodeGenerator(current_node, next_node_index, max_steer_angle, step_size, arc_length, wheel_base, XYbounds, open_space_conf):
        if next_node_index < next_node_num / 2:
            steering = -max_steer_angle + (2 * max_steer_angle / (next_node_num / 2 - 1)) * next_node_index
            traveled_distance = step_size
        else:
            index = next_node_index - next_node_num / 2
            steering = -max_steer_angle + (2 * max_steer_angle / (next_node_num / 2 - 1)) * index
            traveled_distance = -step_size

        intermediate_x, intermediate_y, intermediate_phi = [current_node.x], [current_node.y], [current_node.phi]

        for _ in range(int(arc_length / step_size)):
            next_phi = intermediate_phi[-1] + traveled_distance / wheel_base * math.tan(steering)
            next_x = intermediate_x[-1] + traveled_distance * math.cos((intermediate_phi[-1] + next_phi) / 2)
            next_y = intermediate_y[-1] + traveled_distance * math.sin((intermediate_phi[-1] + next_phi) / 2)
            
            intermediate_x.append(next_x)
            intermediate_y.append(next_y)
            intermediate_phi.append(normalize_angle(next_phi))

        if (intermediate_x[-1] > XYbounds[1] or intermediate_x[-1] < XYbounds[0] or
                intermediate_y[-1] > XYbounds[3] or intermediate_y[-1] < XYbounds[2]):
            return None

        next_node = Node3d(intermediate_x, intermediate_y, intermediate_phi, XYbounds, open_space_conf)
        next_node.SetPre(current_node)
        next_node.SetDirec(traveled_distance > 0)
        next_node.SetSteer(steering)

        return next_node
    
    def LoadRSPinCS(reeds_shepp_to_end, current_node):
        # The equivalent of 'new Node3d(...)' in Python would be simply 'Node3d(...)'
        end_node = Node3d(reeds_shepp_to_end.x, 
                        reeds_shepp_to_end.y, 
                        reeds_shepp_to_end.phi, 
                        XYbounds, 
                        planner_open_space_config)
        end_node.SetPre(current_node)
        end_node.SetTrajCost(current_node.GetTrajCost() + reeds_shepp_to_end.Cost)
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

        # The first {x, y, phi} is collision free unless they are start and end
        # configuration of search problem
        check_start_index = 0 if node_step_size == 1 else 1

        for i in range(check_start_index, node_step_size):
            if (
                traversed_x[i] > self.XYbounds_[1]
                or traversed_x[i] < self.XYbounds_[0]
                or traversed_y[i] > self.XYbounds_[3]
                or traversed_y[i] < self.XYbounds_[2]
            ):
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

def analytic_expansion(current_node):
    reeds_shepp_to_check = ReedSheppPath()
    candidate_final_node = HybridAStar.LoadRSPinCS(reeds_shepp_to_check, current_node)
    return True, candidate_final_node

def plot_dp_map(dp_map, xy_bounds):
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
    # fig, ax = plt.subplots()

    # Plot the cost map as a heatmap
    heatmap = ax.pcolormesh(grid_x, grid_y, cost_map, cmap='viridis')

    # Add a colorbar to show the cost values
    cbar = fig.colorbar(heatmap)
    cbar.set_label('Cost')

    # Set the axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('DP Map')

    # Show the plot
    plt.show()
        
# Initialize plot
fig, ax = plt.subplots()

# Create vehicle boxes
start_box = create_vehicle_box(sx, sy, sphi, vehicle_param["length"], vehicle_param["width"])
end_box = create_vehicle_box(ex, ey, ephi, vehicle_param["length"], vehicle_param["width"])

# Plot vehicle boxes
start_patch = patches.Polygon(start_box, closed=True, edgecolor='green', fill=False)
end_patch = patches.Polygon(end_box, closed=True, edgecolor='red', fill=False)
ax.add_patch(start_patch)
ax.add_patch(end_patch)

# Convert obstacle vertices into line segments and plot them        
obstacles_linesegments_vec: List[List[LineSegment2d]] = []
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
svec_to_center = Vec2d(
    (vehicle_param['front_edge_to_center'] - vehicle_param['back_edge_to_center']) / 2.0,
    (vehicle_param['left_edge_to_center'] - vehicle_param['right_edge_to_center']) / 2.0
)
scenter = sposition + svec_to_center.rotate(sphi)
sbox = Box2d(scenter, sphi, vehicle_param['length'], vehicle_param['width'])

eposition = Vec2d(ex, ey)
evec_to_center = Vec2d(
    (vehicle_param['front_edge_to_center'] - vehicle_param['back_edge_to_center']) / 2.0,
    (vehicle_param['left_edge_to_center'] - vehicle_param['right_edge_to_center']) / 2.0
)
ecenter = eposition + evec_to_center.rotate(ephi)
ebox = Box2d(ecenter, ephi, vehicle_param['length'], vehicle_param['width'])

# Plot vehicle boxes using Node3d.GetBoundingBox
start_node = Node3d([sx], [sy], [sphi], XYbounds)
end_node = Node3d([ex], [ey], [ephi], XYbounds)

hybrid_a_star = HybridAStar(planner_open_space_config, vehicle_param, XYbounds, obstacles_linesegments_vec)
if not hybrid_a_star.ValidityCheck(start_node):
    logging.error("start_node in collision with obstacles")
    logging.error(f"{start_node.GetX()},{start_node.GetY()},{start_node.GetPhi()}")

if not hybrid_a_star.ValidityCheck(end_node):
    logging.error("end_node in collision with obstacles")

start_box_patch = Node3d.GetBoundingBox(vehicle_param, start_node.x, start_node.y, start_node.phi)
end_box_patch = Node3d.GetBoundingBox(vehicle_param, end_node.x, end_node.y, end_node.phi)
# ax.add_patch(start_box_patch)
# ax.add_patch(end_box_patch)

# Generate and plot the DP map as a heatmap
open_pq = []
open_set = {}
grid_search = GridSearch(planner_open_space_config)
dp_map = grid_search.generate_dp_map(ex, ey, XYbounds, obstacles_linesegments_vec, open_pq, open_set)
plot_dp_map(grid_search.dp_map_, XYbounds)
plt.imshow(grid_search.dp_map_, extent=XYbounds, origin='lower', cmap='viridis', alpha=0.5)
plt.colorbar(label='Cost to Goal')

open_set_.add(start_node.GetIndex())
heapq.heappush(open_pq_, (start_node.GetCost(), start_node))
explored_node_num = 0
available_result_num = 0
rs_time = 0
node_generator_time = 0
validity_check_time = 0
heuristic_time = 0
best_explored_num = 0
best_available_result_num = 0
max_explored_num = planner_open_space_config['warm_start_config']['max_explored_num']
desired_explored_num = min(planner_open_space_config['warm_start_config']['desired_explored_num'],
                           planner_open_space_config['warm_start_config']['max_explored_num'])
kMaxNodeNum = 200000
candidate_final_nodes = []
close_set_ = set()
astar_start_time = time()

# Begin Hybrid A* loop
while open_pq_ and len(open_pq_) < kMaxNodeNum and available_result_num < desired_explored_num and explored_node_num < max_explored_num:
    current_node = heapq.heappop(open_pq_)[1]  # Assuming each item is (cost, node)
    rs_start_time = time()

    final_node = None
    success, new_final_node = analytic_expansion(current_node)
    if success:
        if final_node is None or final_node.GetTrajCost() > new_final_node.GetTrajCost():
            final_node = new_final_node
            best_explored_num = explored_node_num + 1
            best_available_result_num = available_result_num + 1
        available_result_num += 1

    explored_node_num += 1
    rs_end_time = time()
    rs_time += rs_end_time - rs_start_time
    close_set_.add(current_node.GetIndex())

    if time() - astar_start_time > planner_open_space_config['warm_start_config']['astar_max_search_time'] and available_result_num > 0:
        break

    next_node_num = 10
    temp_set = set()
    for i in range(next_node_num):
        gen_node_time = time.time()
        next_node = HybridAStar.NextNodeGenerator(current_node, i)  # This function needs to be defined

        if next_node is None or next_node.GetIndex() in close_set_ or not HybridAStar.ValidityCheck(next_node):
            continue
        
        validity_check_time += time.time() - gen_node_time
        
        if next_node.GetIndex() not in open_set_:
            HybridAStar.CalculateNodeCost(current_node, next_node)
            heapq.heappush(open_pq_, (next_node.GetIndex(), next_node))
            temp_set.add(next_node.GetIndex())
    
    open_set_.update(temp_set)

# Plot start and end points
ax.plot(sx, sy, 'go', markersize=10, label='Start')
ax.plot(ex, ey, 'ro', markersize=10, label='End')

# Set the plot bounds
ax.set_xlim(XYbounds[0], XYbounds[1])
ax.set_ylim(XYbounds[2], XYbounds[3])

# Set plot labels and title
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Hybrid A* Planning Visualization')
ax.grid(True)
ax.legend()

plt.show()
plt.show()