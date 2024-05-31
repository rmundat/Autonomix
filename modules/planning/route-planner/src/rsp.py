import math
from typing import List, Tuple

from ML.utils import normalize_angle, cartesian2polar
from ML.config import vehicle_param, planner_open_space_config
from ML.grid import Node3d

class ReedSheppPath:
    def __init__(self):
        self.segs_lengths: List[float] = []
        self.segs_types: List[str] = []
        self.total_length: float = 0.0
        self.cost: float = 0.0
        self.x: List[float] = []
        self.y: List[float] = []
        self.phi: List[float] = []
        self.gear: List[bool] = []  # True for driving forward, False for driving backward
        self.radius: float = 1.0

class RSPParam:
    def __init__(self):
        self.flag: bool = False
        self.t: float = 0.0
        self.u: float = 0.0
        self.v: float = 0.0

class ReedShepp:
    def __init__(self):
        self.vehicle_param_ = vehicle_param
        self.planner_open_space_config_ = planner_open_space_config
        self.max_kappa_ = math.tan(
            self.vehicle_param_["max_steer_angle"]
            * self.planner_open_space_config_["warm_start_config"]["traj_kappa_contraint_ratio"]
            / self.vehicle_param_["steer_ratio"]
        ) / self.vehicle_param_["wheel_base"]
        self.traj_forward_penalty_ = self.planner_open_space_config_["warm_start_config"]["traj_forward_penalty"]
        self.traj_back_penalty_ = self.planner_open_space_config_["warm_start_config"]["traj_back_penalty"]
        self.traj_gear_switch_penalty_ = self.planner_open_space_config_["warm_start_config"]["traj_gear_switch_penalty"]
        self.traj_steer_penalty_ = self.planner_open_space_config_["warm_start_config"]["traj_steer_penalty"]
        self.traj_steer_change_penalty_ = self.planner_open_space_config_["warm_start_config"]["traj_steer_change_penalty"]
        self.traj_short_length_penalty_ = self.planner_open_space_config_["warm_start_config"]["traj_short_length_penalty"]
        self.traj_expected_shortest_length_ = self.planner_open_space_config_["warm_start_config"]["traj_expected_shortest_length"]
        print(f"max kappa: {self.max_kappa_}")
        print(f"traj_short_length_penalty_: {self.traj_short_length_penalty_}")

    def calc_tau_omega(self, u: float, v: float, xi: float, eta: float, phi: float) -> Tuple[float, float]:
        delta = normalize_angle(u - v)
        A = math.sin(u) - math.sin(delta)
        B = math.cos(u) - math.cos(delta) - 1.0

        t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
        t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
        tau = 0.0
        if t2 < 0:
            tau = normalize_angle(t1 + math.pi)
        else:
            tau = normalize_angle(t1)
        omega = normalize_angle(tau - u + v - phi)
        return tau, omega

    def shortest_rsp(self, start_node: Node3d, end_node: Node3d) -> ReedSheppPath:
        optimal_path = ReedSheppPath()
        all_possible_paths = []
        if not self.generate_rsps(start_node, end_node, all_possible_paths):
            print("Fail to generate different combination of Reed Shepp paths")
            return optimal_path

        start_dire = 1
        if not start_node.get_direc():
            start_dire = -1

        min_cost = float('inf')
        optimal_path_index = 0
        for i, path in enumerate(all_possible_paths):
            cost = 0
            steering_radius = path.radius / self.max_kappa_
            steer_change_penalty_cost = math.atan(self.vehicle_param_.wheel_base() / steering_radius * 2.0) * self.traj_steer_change_penalty_
            for j, length in enumerate(path.segs_lengths):
                if path.segs_types[j] != 'S':
                    cost += abs(length) * (self.traj_steer_penalty_) / self.max_kappa_ * path.radius
                    if j > 0 and (path.segs_types[j - 1] != 'S') and (path.segs_types[j - 1] != path.segs_types[j]):
                        cost += steer_change_penalty_cost
                    if abs(length) / self.max_kappa_ * path.radius < self.traj_expected_shortest_length_:
                        cost += self.traj_short_length_penalty_
                else:
                    if length < 0:
                        cost += -length * self.traj_back_penalty_ / self.max_kappa_
                    else:
                        cost += length * self.traj_forward_penalty_ / self.max_kappa_
                    if abs(length) / self.max_kappa_ < self.traj_expected_shortest_length_:
                        cost += self.traj_short_length_penalty_

                if j > 0 and path.segs_lengths[j] * path.segs_lengths[j - 1] < 0:
                    cost += cost + self.traj_gear_switch_penalty_
                if j == 0 and start_dire * path.segs_lengths[j] < 0:
                    cost += cost + self.traj_gear_switch_penalty_

            if cost < min_cost:
                optimal_path_index = i
                min_cost = cost

        if not self.generate_local_configurations(start_node, end_node, all_possible_paths[optimal_path_index]):
            print("Fail to generate local configurations(x, y, phi) in SetRSP")
            return optimal_path

        if abs(all_possible_paths[optimal_path_index].x[-1] - end_node.get_x()) > 1e-3 or \
           abs(all_possible_paths[optimal_path_index].y[-1] - end_node.get_y()) > 1e-3 or \
           normalize_angle(all_possible_paths[optimal_path_index].phi[-1] - end_node.get_phi()) > 1e-3:
            print("RSP end position not right")
            for i, seg_type in enumerate(all_possible_paths[optimal_path_index].segs_types):
                print(f"types are {seg_type}")
            print(f"x, y, phi are: {all_possible_paths[optimal_path_index].x[-1]}, {all_possible_paths[optimal_path_index].y[-1]}, {all_possible_paths[optimal_path_index].phi[-1]}")
            print(f"end x, y, phi are: {end_node.get_x()}, {end_node.get_y()}, {end_node.get_phi()}")
            return optimal_path

        optimal_path.cost = min_cost
        optimal_path.x = all_possible_paths[optimal_path_index].x
        optimal_path.y = all_possible_paths[optimal_path_index].y
        optimal_path.phi = all_possible_paths[optimal_path_index].phi
        optimal_path.gear = all_possible_paths[optimal_path_index].gear
        optimal_path.total_length = all_possible_paths[optimal_path_index].total_length
        optimal_path.segs_types = all_possible_paths[optimal_path_index].segs_types
        optimal_path.segs_lengths = all_possible_paths[optimal_path_index].segs_lengths
        return optimal_path

    def generate_rsps(self, start_node: Node3d, end_node: Node3d, all_possible_paths: List[ReedSheppPath]) -> bool:
        if not self.generate_rsp(start_node, end_node, all_possible_paths):
            print("Fail to generate general profile of different RSPs")
            return False
        return True

    def generate_rsp(self, start_node: Node3d, end_node: Node3d, all_possible_paths: List[ReedSheppPath]) -> bool:
        dx = end_node.get_x() - start_node.get_x()
        dy = end_node.get_y() - start_node.get_y()
        dphi = end_node.get_phi() - start_node.get_phi()
        c = math.cos(start_node.get_phi())
        s = math.sin(start_node.get_phi())
        # normalize the initial point to (0,0,0)
        x = (c * dx + s * dy) * self.max_kappa_
        y = (-s * dx + c * dy) * self.max_kappa_
        if not self.scs(x, y, dphi, all_possible_paths):
            print("Fail at SCS")
        if not self.csc(x, y, dphi, all_possible_paths):
            print("Fail at CSC")
        if not self.ccc(x, y, dphi, all_possible_paths):
            print("Fail at CCC")
        if not self.cccc(x, y, dphi, all_possible_paths):
            print("Fail at CCCC")
        if not self.ccsc(x, y, dphi, all_possible_paths):
            print("Fail at CCSC")
        if not self.ccscc(x, y, dphi, all_possible_paths):
            print("Fail at CCSCC")
        if not all_possible_paths:
            print("No path generated by certain two configurations")
            return False
        return True

    def scs(self, x: float, y: float, phi: float, all_possible_paths: List[ReedSheppPath]) -> bool:
        sls_param = RSPParam()
        self.sls(x, y, phi, sls_param)
        sls_lengths = [sls_param.t, sls_param.u, sls_param.v]
        sls_types = "SLS"
        if sls_param.flag and not self.set_rsp(3, sls_lengths, sls_types, all_possible_paths):
            print("Fail at SetRSP with SLS_param")
            return False

        srs_param = RSPParam()
        self.sls(x, -y, -phi, srs_param)
        srs_lengths = [srs_param.t, srs_param.u, srs_param.v]
        srs_types = "SRS"
        if srs_param.flag and not self.set_rsp(3, srs_lengths, srs_types, all_possible_paths):
            print("Fail at SetRSP with SRS_param")
            return False
        return True

    def cs(self, x: float, y: float, phi: float, all_possible_paths: List[ReedSheppPath]) -> bool:
        ls1_param = RSPParam()
        self.ls(x, y, phi, ls1_param)  # L+S+ L+S-
        ls1_lengths = [ls1_param.t, ls1_param.u]
        ls1_types = "LS"
        if ls1_param.flag and not self.set_rsp(2, ls1_lengths, ls1_types, all_possible_paths, ls1_param.v):
            print("Fail at SetRSP with LS_param")
            return False
        ls2_param = RSPParam()
        self.ls(-x, y, -phi, ls2_param)  # L-S+ L-S-
        ls2_lengths = [-ls2_param.t, -ls2_param.u]
        ls2_types = "LS"
        if ls2_param.flag and not self.set_rsp(2, ls2_lengths, ls2_types, all_possible_paths, ls2_param.v):
            print("Fail at SetRSP with LS_param")
            return False
        rs1_param = RSPParam()
        self.ls(x, -y, -phi, rs1_param)  # R+S+ R+S-
        rs1_lengths = [rs1_param.t, rs1_param.u]
        rs1_types = "RS"
        if rs1_param.flag and not self.set_rsp(2, rs1_lengths, rs1_types, all_possible_paths, rs1_param.v):
            print("Fail at SetRSP with SLS_param")
            return False

        rs2_param = RSPParam()
        self.ls(-x, -y, phi, rs2_param)  # R-S+ R-S-
        rs2_lengths = [-rs2_param.t, -rs2_param.u]
        rs2_types = "RS"
        if rs2_param.flag and not self.set_rsp(2, rs2_lengths, rs2_types, all_possible_paths, rs2_param.v):
            print("Fail at SetRSP with SLS_param")
            return False
        return True

    def csc(self, x: float, y: float, phi: float, all_possible_paths: List[ReedSheppPath]) -> bool:
        lsl1_param = RSPParam()
        self.lsl(x, y, phi, lsl1_param)
        lsl1_lengths = [lsl1_param.t, lsl1_param.u, lsl1_param.v]
        lsl1_types = "LSL"
        if lsl1_param.flag and not self.set_rsp(3, lsl1_lengths, lsl1_types, all_possible_paths):
            print("Fail at SetRSP with LSL_param")
            return False

        lsl2_param = RSPParam()
        self.lsl(-x, y, -phi, lsl2_param)
        lsl2_lengths = [-lsl2_param.t, -lsl2_param.u, -lsl2_param.v]
        lsl2_types = "LSL"
        if lsl2_param.flag and not self.set_rsp(3, lsl2_lengths, lsl2_types, all_possible_paths):
            print("Fail at SetRSP with LSL2_param")
            return False

        lsl3_param = RSPParam()
        self.lsl(x, -y, -phi, lsl3_param)
        lsl3_lengths = [lsl3_param.t, lsl3_param.u, lsl3_param.v]
        lsl3_types = "RSR"
        if lsl3_param.flag and not self.set_rsp(3, lsl3_lengths, lsl3_types, all_possible_paths):
            print("Fail at SetRSP with LSL3_param")
            return False

        lsl4_param = RSPParam()
        self.lsl(-x, -y, phi, lsl4_param)
        lsl4_lengths = [-lsl4_param.t, -lsl4_param.u, -lsl4_param.v]
        lsl4_types = "RSR"
        if lsl4_param.flag and not self.set_rsp(3, lsl4_lengths, lsl4_types, all_possible_paths):
            print("Fail at SetRSP with LSL4_param")
            return False
        
        lsr1_param = RSPParam()
        self.lsr(x, y, phi, lsr1_param)
        lsr1_lengths = [lsr1_param.t, lsr1_param.u, lsr1_param.v]
        lsr1_types = "LSR"
        if lsr1_param.flag and not self.set_rsp(3, lsr1_lengths, lsr1_types, all_possible_paths):
            print("Fail at SetRSP with LSR1_param")
            return False

        lsr2_param = RSPParam()
        self.lsr(-x, y, -phi, lsr2_param)
        lsr2_lengths = [-lsr2_param.t, -lsr2_param.u, -lsr2_param.v]
        lsr2_types = "LSR"
        if lsr2_param.flag and not self.set_rsp(3, lsr2_lengths, lsr2_types, all_possible_paths):
            print("Fail at SetRSP with LSR2_param")
            return False

        lsr3_param = RSPParam()
        self.lsr(x, -y, -phi, lsr3_param)
        lsr3_lengths = [lsr3_param.t, lsr3_param.u, lsr3_param.v]
        lsr3_types = "RSL"
        if lsr3_param.flag and not self.set_rsp(3, lsr3_lengths, lsr3_types, all_possible_paths):
            print("Fail at SetRSP with LSR3_param")
            return False

        lsr4_param = RSPParam()
        self.lsr(-x, -y, phi, lsr4_param)
        lsr4_lengths = [-lsr4_param.t, -lsr4_param.u, -lsr4_param.v]
        lsr4_types = "RSL"
        if lsr4_param.flag and not self.set_rsp(3, lsr4_lengths, lsr4_types, all_possible_paths):
            print("Fail at SetRSP with LSR4_param")
            return False
        return True

    def ccc(self, x: float, y: float, phi: float, all_possible_paths: List[ReedSheppPath]) -> bool:
        lrl1_param = RSPParam()
        self.lrl(x, y, phi, lrl1_param)
        lrl1_lengths = [lrl1_param.t, lrl1_param.u, lrl1_param.v]
        lrl1_types = "LRL"
        if lrl1_param.flag and not self.set_rsp(3, lrl1_lengths, lrl1_types, all_possible_paths):
            print("Fail at SetRSP with LRL_param")
            return False

        lrl2_param = RSPParam()
        self.lrl(-x, y, -phi, lrl2_param)
        lrl2_lengths = [-lrl2_param.t, -lrl2_param.u, -lrl2_param.v]
        lrl2_types = "LRL"
        if lrl2_param.flag and not self.set_rsp(3, lrl2_lengths, lrl2_types, all_possible_paths):
            print("Fail at SetRSP with LRL2_param")
            return False

        lrl3_param = RSPParam()
        self.lrl(x, -y, -phi, lrl3_param)
        lrl3_lengths = [lrl3_param.t, lrl3_param.u, lrl3_param.v]
        lrl3_types = "RLR"
        if lrl3_param.flag and not self.set_rsp(3, lrl3_lengths, lrl3_types, all_possible_paths):
            print("Fail at SetRSP with LRL3_param")
            return False

        lrl4_param = RSPParam()
        self.lrl(-x, -y, phi, lrl4_param)
        lrl4_lengths = [-lrl4_param.t, -lrl4_param.u, -lrl4_param.v]
        lrl4_types = "RLR"
        if lrl4_param.flag and not self.set_rsp(3, lrl4_lengths, lrl4_types, all_possible_paths):
            print("Fail at SetRSP with LRL4_param")
            return False

        # backward
        xb = x * math.cos(phi) + y * math.sin(phi)
        yb = x * math.sin(phi) - y * math.cos(phi)

        lrl5_param = RSPParam()
        self.lrl(xb, yb, phi, lrl5_param)
        lrl5_lengths = [lrl5_param.v, lrl5_param.u, lrl5_param.t]
        lrl5_types = "LRL"
        if lrl5_param.flag and not self.set_rsp(3, lrl5_lengths, lrl5_types, all_possible_paths):
            print("Fail at SetRSP with LRL5_param")
            return False

        lrl6_param = RSPParam()
        self.lrl(-xb, yb, -phi, lrl6_param)
        lrl6_lengths = [-lrl6_param.v, -lrl6_param.u, -lrl6_param.t]
        lrl6_types = "LRL"
        if lrl6_param.flag and not self.set_rsp(3, lrl6_lengths, lrl6_types, all_possible_paths):
            print("Fail at SetRSP with LRL6_param")
            return False

        lrl7_param = RSPParam()
        self.lrl(xb, -yb, -phi, lrl7_param)
        lrl7_lengths = [lrl7_param.v, lrl7_param.u, lrl7_param.t]
        lrl7_types = "RLR"
        if lrl7_param.flag and not self.set_rsp(3, lrl7_lengths, lrl7_types, all_possible_paths):
            print("Fail at SetRSP with LRL7_param")
            return False

        lrl8_param = RSPParam()
        self.lrl(-xb, -yb, phi, lrl8_param)
        lrl8_lengths = [-lrl8_param.v, -lrl8_param.u, -lrl8_param.t]
        lrl8_types = "RLR"
        if lrl8_param.flag and not self.set_rsp(3, lrl8_lengths, lrl8_types, all_possible_paths):
            print("Fail at SetRSP with LRL8_param")
            return False
        return True

    def cccc(self, x: float, y: float, phi: float, all_possible_paths: List[ReedSheppPath]) -> bool:
        lrlrn1_param = RSPParam()
        self.lrlrn(x, y, phi, lrlrn1_param)
        lrlrn1_lengths = [lrlrn1_param.t, lrlrn1_param.u, -lrlrn1_param.u, lrlrn1_param.v]
        lrlrn1_types = "LRLR"
        if lrlrn1_param.flag and not self.set_rsp(4, lrlrn1_lengths, lrlrn1_types, all_possible_paths):
            print("Fail at SetRSP with LRLRn_param")
            return False

        lrlrn2_param = RSPParam()
        self.lrlrn(-x, y, -phi, lrlrn2_param)
        lrlrn2_lengths = [-lrlrn2_param.t, -lrlrn2_param.u, lrlrn2_param.u, -lrlrn2_param.v]
        lrlrn2_types = "LRLR"
        if lrlrn2_param.flag and not self.set_rsp(4, lrlrn2_lengths, lrlrn2_types, all_possible_paths):
            print("Fail at SetRSP with LRLRn2_param")
            return False

        lrlrn3_param = RSPParam()
        self.lrlrn(x, -y, -phi, lrlrn3_param)
        lrlrn3_lengths = [lrlrn3_param.t, lrlrn3_param.u, -lrlrn3_param.u, lrlrn3_param.v]
        lrlrn3_types = "RLRL"
        if lrlrn3_param.flag and not self.set_rsp(4, lrlrn3_lengths, lrlrn3_types, all_possible_paths):
            print("Fail at SetRSP with LRLRn3_param")
            return False

        lrlrn4_param = RSPParam()
        self.lrlrn(-x, -y, phi, lrlrn4_param)
        lrlrn4_lengths = [-lrlrn4_param.t, -lrlrn4_param.u, lrlrn4_param.u, -lrlrn4_param.v]
        lrlrn4_types = "RLRL"
        if lrlrn4_param.flag and not self.set_rsp(4, lrlrn4_lengths, lrlrn4_types, all_possible_paths):
            print("Fail at SetRSP with LRLRn4_param")
            return False

        lrlrp1_param = RSPParam()
        self.lrlrp(x, y, phi, lrlrp1_param)
        lrlrp1_lengths = [lrlrp1_param.t, lrlrp1_param.u, lrlrp1_param.u, lrlrp1_param.v]
        lrlrp1_types = "LRLR"
        if lrlrp1_param.flag and not self.set_rsp(4, lrlrp1_lengths, lrlrp1_types, all_possible_paths):
            print("Fail at SetRSP with LRLRp1_param")
            return False

        lrlrp2_param = RSPParam()
        self.lrlrp(-x, y, -phi, lrlrp2_param)
        lrlrp2_lengths = [-lrlrp2_param.t, -lrlrp2_param.u, -lrlrp2_param.u, -lrlrp2_param.v]
        lrlrp2_types = "LRLR"
        if lrlrp2_param.flag and not self.set_rsp(4, lrlrp2_lengths, lrlrp2_types, all_possible_paths):
            print("Fail at SetRSP with LRLRp2_param")
            return False

        lrlrp3_param = RSPParam()
        self.lrlrp(x, -y, -phi, lrlrp3_param)
        lrlrp3_lengths = [lrlrp3_param.t, lrlrp3_param.u, lrlrp3_param.u, lrlrp3_param.v]
        lrlrp3_types = "RLRL"
        if lrlrp3_param.flag and not self.set_rsp(4, lrlrp3_lengths, lrlrp3_types, all_possible_paths):
            print("Fail at SetRSP with LRLRp3_param")
            return False

        lrlrp4_param = RSPParam()
        self.lrlrp(-x, -y, phi, lrlrp4_param)
        lrlrp4_lengths = [-lrlrp4_param.t, -lrlrp4_param.u, -lrlrp4_param.u, -lrlrp4_param.v]
        lrlrp4_types = "RLRL"
        if lrlrp4_param.flag and not self.set_rsp(4, lrlrp4_lengths, lrlrp4_types, all_possible_paths):
            print("Fail at SetRSP with LRLRp4_param")
            return False
        return True

    def ccsc(self, x: float, y: float, phi: float, all_possible_paths: List[ReedSheppPath]) -> bool:
        lrsl1_param = RSPParam()
        self.lrsl(x, y, phi, lrsl1_param)
        lrsl1_lengths = [lrsl1_param.t, -0.5 * math.pi, lrsl1_param.u, lrsl1_param.v]
        lrsl1_types = "LRSL"
        if lrsl1_param.flag and not self.set_rsp(4, lrsl1_lengths, lrsl1_types, all_possible_paths):
            print("Fail at SetRSP with LRSL1_param")
            return False

        lrsl2_param = RSPParam()
        self.lrsl(-x, y, -phi, lrsl2_param)
        lrsl2_lengths = [-lrsl2_param.t, 0.5 * math.pi, -lrsl2_param.u, -lrsl2_param.v]
        lrsl2_types = "LRSL"
        if lrsl2_param.flag and not self.set_rsp(4, lrsl2_lengths, lrsl2_types, all_possible_paths):
            print("Fail at SetRSP with LRSL2_param")
            return False

        lrsl3_param = RSPParam()
        self.lrsl(x, -y, -phi, lrsl3_param)
        lrsl3_lengths = [lrsl3_param.t, -0.5 * math.pi, lrsl3_param.u, lrsl3_param.v]
        lrsl3_types = "RLSR"
        if lrsl3_param.flag and not self.set_rsp(4, lrsl3_lengths, lrsl3_types, all_possible_paths):
            print("Fail at SetRSP with LRSL3_param")
            return False

        lrsl4_param = RSPParam()
        self.lrsl(-x, -y, phi, lrsl4_param)
        lrsl4_lengths = [-lrsl4_param.t, 0.5 * math.pi, -lrsl4_param.u, -lrsl4_param.v]
        lrsl4_types = "RLSR"
        if lrsl4_param.flag and not self.set_rsp(4, lrsl4_lengths, lrsl4_types, all_possible_paths):
            print("Fail at SetRSP with LRSL4_param")
            return False

        lrsr1_param = RSPParam()
        self.lrsr(x, y, phi, lrsr1_param)
        lrsr1_lengths = [lrsr1_param.t, -0.5 * math.pi, lrsr1_param.u, lrsr1_param.v]
        lrsr1_types = "LRSR"
        if lrsr1_param.flag and not self.set_rsp(4, lrsr1_lengths, lrsr1_types, all_possible_paths):
            print("Fail at SetRSP with LRSR1_param")
            return False
        
        lrsr2_param = RSPParam()
        self.lrsr(-x, y, -phi, lrsr2_param)
        lrsr2_lengths = [-lrsr2_param.t, 0.5 * math.pi, -lrsr2_param.u, -lrsr2_param.v]
        lrsr2_types = "LRSR"
        if lrsr2_param.flag and not self.set_rsp(4, lrsr2_lengths, lrsr2_types, all_possible_paths):
            print("Fail at SetRSP with LRSR2_param")
            return False

        lrsr3_param = RSPParam()
        self.lrsr(x, -y, -phi, lrsr3_param)
        lrsr3_lengths = [lrsr3_param.t, -0.5 * math.pi, lrsr3_param.u, lrsr3_param.v]
        lrsr3_types = "RLSL"
        if lrsr3_param.flag and not self.set_rsp(4, lrsr3_lengths, lrsr3_types, all_possible_paths):
            print("Fail at SetRSP with LRSR3_param")
            return False

        lrsr4_param = RSPParam()
        self.lrsr(-x, -y, phi, lrsr4_param)
        lrsr4_lengths = [-lrsr4_param.t, 0.5 * math.pi, -lrsr4_param.u, -lrsr4_param.v]
        lrsr4_types = "RLSL"
        if lrsr4_param.flag and not self.set_rsp(4, lrsr4_lengths, lrsr4_types, all_possible_paths):
            print("Fail at SetRSP with LRSR4_param")
            return False

        # backward
        xb = x * math.cos(phi) + y * math.sin(phi)
        yb = x * math.sin(phi) - y * math.cos(phi)

        lrsl5_param = RSPParam()
        self.lrsl(xb, yb, phi, lrsl5_param)
        lrsl5_lengths = [lrsl5_param.v, lrsl5_param.u, -0.5 * math.pi, lrsl5_param.t]
        lrsl5_types = "LSRL"
        if lrsl5_param.flag and not self.set_rsp(4, lrsl5_lengths, lrsl5_types, all_possible_paths):
            print("Fail at SetRSP with LRLRn_param")
            return False

        lrsl6_param = RSPParam()
        self.lrsl(-xb, yb, -phi, lrsl6_param)
        lrsl6_lengths = [-lrsl6_param.v, -lrsl6_param.u, 0.5 * math.pi, -lrsl6_param.t]
        lrsl6_types = "LSRL"
        if lrsl6_param.flag and not self.set_rsp(4, lrsl6_lengths, lrsl6_types, all_possible_paths):
            print("Fail at SetRSP with LRSL6_param")
            return False

        lrsl7_param = RSPParam()
        self.lrsl(xb, -yb, -phi, lrsl7_param)
        lrsl7_lengths = [lrsl7_param.v, lrsl7_param.u, -0.5 * math.pi, lrsl7_param.t]
        lrsl7_types = "RSLR"
        if lrsl7_param.flag and not self.set_rsp(4, lrsl7_lengths, lrsl7_types, all_possible_paths):
            print("Fail at SetRSP with LRSL7_param")
            return False

        lrsl8_param = RSPParam()
        self.lrsl(-xb, -yb, phi, lrsl8_param)
        lrsl8_lengths = [-lrsl8_param.v, -lrsl8_param.u, 0.5 * math.pi, -lrsl8_param.t]
        lrsl8_types = "RSLR"
        if lrsl8_param.flag and not self.set_rsp(4, lrsl8_lengths, lrsl8_types, all_possible_paths):
            print("Fail at SetRSP with LRSL8_param")
            return False

        lrsr5_param = RSPParam()
        self.lrsr(xb, yb, phi, lrsr5_param)
        lrsr5_lengths = [lrsr5_param.v, lrsr5_param.u, -0.5 * math.pi, lrsr5_param.t]
        lrsr5_types = "RSRL"
        if lrsr5_param.flag and not self.set_rsp(4, lrsr5_lengths, lrsr5_types, all_possible_paths):
            print("Fail at SetRSP with LRSR5_param")
            return False

        lrsr6_param = RSPParam()
        self.lrsr(-xb, yb, -phi, lrsr6_param)
        lrsr6_lengths = [-lrsr6_param.v, -lrsr6_param.u, 0.5 * math.pi, -lrsr6_param.t]
        lrsr6_types = "RSRL"
        if lrsr6_param.flag and not self.set_rsp(4, lrsr6_lengths, lrsr6_types, all_possible_paths):
            print("Fail at SetRSP with LRSR6_param")
            return False

        lrsr7_param = RSPParam()
        self.lrsr(xb, -yb, -phi, lrsr7_param)
        lrsr7_lengths = [lrsr7_param.v, lrsr7_param.u, -0.5 * math.pi, lrsr7_param.t]
        lrsr7_types = "LSLR"
        if lrsr7_param.flag and not self.set_rsp(4, lrsr7_lengths, lrsr7_types, all_possible_paths):
            print("Fail at SetRSP with LRSR7_param")
            return False

        lrsr8_param = RSPParam()
        self.lrsr(-xb, -yb, phi, lrsr8_param)
        lrsr8_lengths = [-lrsr8_param.v, -lrsr8_param.u, 0.5 * math.pi, -lrsr8_param.t]
        lrsr8_types = "LSLR"
        if lrsr8_param.flag and not self.set_rsp(4, lrsr8_lengths, lrsr8_types, all_possible_paths):
            print("Fail at SetRSP with LRSR8_param")
            return False
        return True

    def ccscc(self, x: float, y: float, phi: float, all_possible_paths: List[ReedSheppPath]) -> bool:
        lrslr1_param = RSPParam()
        self.lrslr(x, y, phi, lrslr1_param)
        lrslr1_lengths = [lrslr1_param.t, -0.5 * math.pi, lrslr1_param.u, -0.5 * math.pi, lrslr1_param.v]
        lrslr1_types = "LRSLR"
        if lrslr1_param.flag and not self.set_rsp(5, lrslr1_lengths, lrslr1_types, all_possible_paths):
            print("Fail at SetRSP with LRSLR1_param")
            return False

        lrslr2_param = RSPParam()
        self.lrslr(-x, y, -phi, lrslr2_param)
        lrslr2_lengths = [-lrslr2_param.t, 0.5 * math.pi, -lrslr2_param.u, 0.5 * math.pi, -lrslr2_param.v]
        lrslr2_types = "LRSLR"
        if lrslr2_param.flag and not self.set_rsp(5, lrslr2_lengths, lrslr2_types, all_possible_paths):
            print("Fail at SetRSP with LRSLR2_param")
            return False

        lrslr3_param = RSPParam()
        self.lrslr(x, -y, -phi, lrslr3_param)
        lrslr3_lengths = [lrslr3_param.t, -0.5 * math.pi, lrslr3_param.u, -0.5 * math.pi, lrslr3_param.v]
        lrslr3_types = "RLSRL"
        if lrslr3_param.flag and not self.set_rsp(5, lrslr3_lengths, lrslr3_types, all_possible_paths):
            print("Fail at SetRSP with LRSLR3_param")
            return False

        lrslr4_param = RSPParam()
        self.lrslr(-x, -y, phi, lrslr4_param)
        lrslr4_lengths = [-lrslr4_param.t, 0.5 * math.pi, -lrslr4_param.u, 0.5 * math.pi, -lrslr4_param.v]
        lrslr4_types = "RLSRL"
        if lrslr4_param.flag and not self.set_rsp(5, lrslr4_lengths, lrslr4_types, all_possible_paths):
            print("Fail at SetRSP with LRSLR4_param")
            return False
        return True

    def lsl(self, x: float, y: float, phi: float, param: RSPParam):
        polar = cartesian2polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
        u = polar[0]
        t = polar[1]
        v = 0.0
        if t >= 0.0:
            v = normalize_angle(phi - t)
            if v >= 0.0:
                param.flag = True
                param.u = u
                param.t = t
                param.v = v

    def lsr(self, x: float, y: float, phi: float, param: RSPParam):
        polar = cartesian2polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
        u1 = polar[0] * polar[0]
        t1 = polar[1]
        u = 0.0
        theta = 0.0
        t = 0.0
        v = 0.0
        if u1 >= 4.0:
            u = math.sqrt(u1 - 4.0)
            theta = math.atan2(2.0, u)
            t = normalize_angle(t1 + theta)
            v = normalize_angle(t - phi)
            if t >= 0.0 and v >= 0.0:
                param.flag = True
                param.u = u
                param.t = t
                param.v = v

    def lrl(self, x: float, y: float, phi: float, param: RSPParam):
        polar = cartesian2polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
        u1 = polar[0]
        t1 = polar[1]
        u = 0.0
        t = 0.0
        v = 0.0
        if u1 <= 4.0:
            u = -2.0 * math.asin(0.25 * u1)
            t = normalize_angle(t1 + 0.5 * u + math.pi)
            v = normalize_angle(phi - t + u)
            if t >= 0.0 and u <= 0.0:
                param.flag = True
                param.u = u
                param.t = t
                param.v = v

    def ls(self, x: float, y: float, phi: float, param: RSPParam):
        phi_mod = normalize_angle(phi)
        epsilon = 1e-6
        if abs(phi_mod - math.pi / 2) < epsilon or abs(phi_mod) < epsilon or \
        abs(phi_mod + math.pi / 2) < epsilon or abs(phi_mod + math.pi) < epsilon:
            return
        r = (y - x * math.tan(phi_mod)) / (1 - 1 / math.cos(phi_mod))
        if r < 1.0:
            return
        # u>0 :lpsp ,u<0 :lpsn
        u = x / math.cos(phi_mod) - r * math.tan(phi_mod)
        t = phi_mod
        if phi_mod < 0:
            t = phi_mod + 2 * math.pi
        param.flag = True
        param.u = u
        param.t = t
        param.v = r

    def sls(self, x: float, y: float, phi: float, param: RSPParam):
        phi_mod = normalize_angle(phi)
        xd = 0.0
        u = 0.0
        t = 0.0
        v = 0.0
        epsilon = 1e-1
        if y > 0.0 and phi_mod > epsilon and phi_mod < math.pi:
            xd = -y / math.tan(phi_mod) + x
            t = xd - math.tan(phi_mod / 2.0)
            u = phi_mod
            v = math.sqrt((x - xd) * (x - xd) + y * y) - math.tan(phi_mod / 2.0)
            param.flag = True
            param.u = u
            param.t = t
            param.v = v
        elif y < 0.0 and phi_mod > epsilon and phi_mod < math.pi:
            xd = -y / math.tan(phi_mod) + x
            t = xd - math.tan(phi_mod / 2.0)
            u = phi_mod
            v = -math.sqrt((x - xd) * (x - xd) + y * y) - math.tan(phi_mod / 2.0)
            param.flag = True
            param.u = u
            param.t = t
            param.v = v

    def lrlrn(self, x: float, y: float, phi: float, param: RSPParam):
        xi = x + math.sin(phi)
        eta = y - 1.0 - math.cos(phi)
        rho = 0.25 * (2.0 + math.sqrt(xi * xi + eta * eta))
        u = 0.0
        if 0.0 <= rho <= 1.0:
            u = math.acos(rho)
            if 0.0 <= u <= 0.5 * math.pi:
                tau_omega = self.calc_tau_omega(u, -u, xi, eta, phi)
                if tau_omega[0] >= 0.0 and tau_omega[1] <= 0.0:
                    param.flag = True
                    param.u = u
                    param.t = tau_omega[0]
                    param.v = tau_omega[1]
                    
    def lrlrp(self, x: float, y: float, phi: float, param: RSPParam):
        xi = x + math.sin(phi)
        eta = y - 1.0 - math.cos(phi)
        rho = (20.0 - xi * xi - eta * eta) / 16.0
        u = 0.0
        if 0.0 <= rho <= 1.0:
            u = -math.acos(rho)
            if 0.0 <= u <= 0.5 * math.pi:
                tau_omega = self.calc_tau_omega(u, u, xi, eta, phi)
                if tau_omega[0] >= 0.0 and tau_omega[1] >= 0.0:
                    param.flag = True
                    param.u = u
                    param.t = tau_omega[0]
                    param.v = tau_omega[1]

    def lrsr(self, x: float, y: float, phi: float, param: RSPParam):
        xi = x + math.sin(phi)
        eta = y - 1.0 - math.cos(phi)
        polar = cartesian2polar(-eta, xi)
        rho = polar[0]
        theta = polar[1]
        t = 0.0
        u = 0.0
        v = 0.0
        if rho >= 2.0:
            t = theta
            u = 2.0 - rho
            v = normalize_angle(t + 0.5 * math.pi - phi)
            if t >= 0.0 and u <= 0.0 and v <= 0.0:
                param.flag = True
                param.u = u
                param.t = t
                param.v = v

    def lrsl(self, x: float, y: float, phi: float, param: RSPParam):
        xi = x - math.sin(phi)
        eta = y - 1.0 + math.cos(phi)
        polar = cartesian2polar(xi, eta)
        rho = polar[0]
        theta = polar[1]
        r = 0.0
        t = 0.0
        u = 0.0
        v = 0.0

        if rho >= 2.0:
            r = math.sqrt(rho * rho - 4.0)
            u = 2.0 - r
            t = normalize_angle(theta + math.atan2(r, -2.0))
            v = normalize_angle(phi - 0.5 * math.pi - t)
            if t >= 0.0 and u <= 0.0 and v <= 0.0:
                param.flag = True
                param.u = u
                param.t = t
                param.v = v

    def lrslr(self, x: float, y: float, phi: float, param: RSPParam):
        xi = x + math.sin(phi)
        eta = y - 1.0 - math.cos(phi)
        polar = cartesian2polar(xi, eta)
        rho = polar[0]
        t = 0.0
        u = 0.0
        v = 0.0
        if rho >= 2.0:
            u = 4.0 - math.sqrt(rho * rho - 4.0)
            if u <= 0.0:
                t = normalize_angle(
                    math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
                v = normalize_angle(t - phi)

                if t >= 0.0 and v >= 0.0:
                    param.flag = True
                    param.u = u
                    param.t = t
                    param.v = v

    def set_rsp(self, size: int, lengths: List[float], types: str, all_possible_paths: List[ReedSheppPath],
                radius: float = 1.0):
        path = ReedSheppPath()
        path.segs_lengths = lengths[:size]
        path.segs_types = list(types[:size])
        path.radius = radius
        sum_length = sum(map(abs, lengths[:size]))
        path.total_length = sum_length
        if path.total_length <= 0.0:
            print("total length smaller than 0")
            return False
        all_possible_paths.append(path)
        return True

    def generate_local_configurations(self, start_node: Node3d, end_node: Node3d,
                                    shortest_path: ReedSheppPath) -> bool:
        radius = shortest_path.radius
        step_scaled = self.planner_open_space_config_.warm_start_config().step_size() * self.max_kappa_
        print(f"shortest_path->total_length {shortest_path.total_length}")
        point_num = int(math.floor(shortest_path.total_length / step_scaled * radius +
                                len(shortest_path.segs_lengths) + 4))
        print(f"point_num {point_num}")
        px = [0.0] * point_num
        py = [0.0] * point_num
        pphi = [0.0] * point_num
        pgear = [True] * point_num
        index = 1
        d = 0.0
        pd = 0.0
        ll = 0.0

        if shortest_path.segs_lengths[0] > 0.0:
            pgear[0] = True
            d = step_scaled
        else:
            pgear[0] = False
            d = -step_scaled
        pd = d
        for i, (m, l) in enumerate(zip(shortest_path.segs_types, shortest_path.segs_lengths)):
            d = step_scaled if l > 0.0 else -step_scaled
            d = d if m == 'S' else d / radius
            ox = px[index]
            oy = py[index]
            ophi = pphi[index]
            index -= 1
            if i >= 1 and shortest_path.segs_lengths[i - 1] * shortest_path.segs_lengths[i] > 0:
                pd = -d - ll
            else:
                pd = d - ll
            while abs(pd) <= abs(l):
                index += 1
                self.interpolation(index, pd, m, ox, oy, ophi, radius, px, py, pphi, pgear)
                pd += d
            ll = l - pd - d
            index += 1
            self.interpolation(index, l, m, ox, oy, ophi, radius, px, py, pphi, pgear)

        epsilon = 1e-15
        while abs(px[-1]) < epsilon and abs(py[-1]) < epsilon and abs(pphi[-1]) < epsilon and pgear[-1]:
            px.pop()
            py.pop()
            pphi.pop()
            pgear.pop()

        for i in range(len(px)):
            shortest_path.x.append(math.cos(-start_node.get_phi()) * px[i] +
                                math.sin(-start_node.get_phi()) * py[i] +
                                start_node.get_x())
            shortest_path.y.append(-math.sin(-start_node.get_phi()) * px[i] +
                                math.cos(-start_node.get_phi()) * py[i] +
                                start_node.get_y())
            shortest_path.phi.append(normalize_angle(pphi[i] + start_node.get_phi()))

        shortest_path.gear = pgear
        for i in range(len(shortest_path.segs_lengths)):
            shortest_path.segs_lengths[i] = shortest_path.segs_lengths[i] / self.max_kappa_
        shortest_path.total_length = shortest_path.total_length / self.max_kappa_
        return True

    def interpolation(self, index: int, pd: float, m: str, ox: float, oy: float, ophi: float, radius: float,
                    px: List[float], py: List[float], pphi: List[float], pgear: List[bool]):
        ldx = 0.0
        ldy = 0.0
        gdx = 0.0
        gdy = 0.0
        if m == 'S':
            px[index] = ox + pd / self.max_kappa_ * math.cos(ophi)
            py[index] = oy + pd / self.max_kappa_ * math.sin(ophi)
            pphi[index] = ophi
        else:
            ldx = math.sin(pd) / self.max_kappa_ * radius
            if m == 'L':
                ldy = (1.0 - math.cos(pd)) / self.max_kappa_ * radius
            elif m == 'R':
                ldy = (1.0 - math.cos(pd)) / -self.max_kappa_ * radius
            gdx = math.cos(-ophi) * ldx + math.sin(-ophi) * ldy
            gdy = -math.sin(-ophi) * ldx + math.cos(-ophi) * ldy
            px[index] = ox + gdx
            py[index] = oy + gdy

        pgear[index] = True if pd > 0.0 else False

        if m == 'L':
            pphi[index] = ophi + pd
        elif m == 'R':
            pphi[index] = ophi - pd