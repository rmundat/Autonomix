import math
from typing import Tuple
import numpy as np

kMathEpsilon = 1e-10

def sqr(x: float) -> float:
    return x * x

def cross_prod(start_point: np.ndarray, end_point_1: np.ndarray, end_point_2: np.ndarray) -> float:
    return np.cross(end_point_1 - start_point, end_point_2 - start_point)

def inner_prod(start_point: np.ndarray, end_point_1: np.ndarray, end_point_2: np.ndarray) -> float:
    return np.dot(end_point_1 - start_point, end_point_2 - start_point)

def cross_prod(x0: float, y0: float, x1: float, y1: float) -> float:
    return x0 * y1 - x1 * y0

def inner_prod(x0: float, y0: float, x1: float, y1: float) -> float:
    return x0 * x1 + y0 * y1

def wrap_angle(angle: float) -> float:
    new_angle = angle % (2.0 * math.pi)
    return new_angle if new_angle >= 0 else new_angle + 2.0 * math.pi

def normalize_angle(angle: float) -> float:
    a = (angle + math.pi) % (2.0 * math.pi)
    return a - math.pi

def angle_diff(from_angle: float, to_angle: float) -> float:
    return normalize_angle(to_angle - from_angle)

def random_int(s: int, t: int, rand_seed: int = 1) -> int:
    if s >= t:
        return s
    import random
    random.seed(rand_seed)
    return random.randint(s, t)

def random_double(s: float, t: float, rand_seed: int = 1) -> float:
    import random
    random.seed(rand_seed)
    return s + (t - s) * random.random()

def gaussian(u: float, std: float, x: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * std * std)) * math.exp(-(x - u) * (x - u) / (2 * std * std))

def rotate_vector2d(v_in: np.ndarray, theta: float) -> np.ndarray:
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x = cos_theta * v_in[0] - sin_theta * v_in[1]
    y = sin_theta * v_in[0] + cos_theta * v_in[1]
    return np.array([x, y])

def cartesian2polar(x: float, y: float) -> Tuple[float, float]:
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r, theta

def check_negative(input_data: float) -> float:
    if input_data < 0:
        input_data = -input_data
    return input_data

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def rfu_to_flu(x: float, y: float) -> Tuple[float, float]:
    return y, -x

def flu_to_rfu(x: float, y: float) -> Tuple[float, float]:
    return -y, x

def l2_norm(feat_dim: int, feat_data: np.ndarray) -> None:
    if feat_dim == 0:
        return
    l2norm = np.linalg.norm(feat_data)
    if l2norm == 0:
        val = 1.0 / math.sqrt(feat_dim)
        feat_data[:] = val
    else:
        feat_data /= l2norm

def almost_equal(x: float, y: float, ulp: int = 2) -> bool:
    return abs(x - y) <= max(1e-8, ulp * math.ldexp(0.5, -53)) or abs(x - y) < 1e-8

class Vec2d:
    def __init__(self, x=0.0, y=0.0):
        self.x_ = x
        self.y_ = y

    def x(self):
        return self.x_

    def y(self):
        return self.y_

    def set_x(self, x):
        self.x_ = x

    def set_y(self, y):
        self.y_ = y

    def length(self):
        return math.sqrt(self.x_ ** 2 + self.y_ ** 2)

    def cross_prod(self, other):
        return self.x_ * other.y() - self.y_ * other.x()

    def inner_prod(self, other):
        return self.x_ * other.x() + self.y_ * other.y()
    
    def rotate(self, angle):
        return Vec2d(self.x_ * math.cos(angle) - self.y_ * math.sin(angle),
               self.x_ * math.sin(angle) + self.y_ * math.cos(angle))
        
    def __add__(self, other):
        return Vec2d(self.x_ + other.x_, self.y_ + other.y_)
            
    @staticmethod
    def create_unit_vec2d(angle):
        return Vec2d(math.cos(angle), math.sin(angle))

    def distance_to(self, other):
        return math.hypot(self.x_ - other.x_, self.y_ - other.y_)

class LineSegment2d:
    def __init__(self, start=None, end=None):
        if start is None and end is None:
            self.start_ = Vec2d()
            self.end_ = Vec2d()
            self.unit_direction_ = Vec2d(1, 0)
        else:
            self.start_ = start
            self.end_ = end
            dx = end.x_ - start.x_
            dy = end.y_ - start.y_
            self.length_ = math.hypot(dx, dy)
            if self.length_ <= kMathEpsilon:
                self.unit_direction_ = Vec2d(0, 0)
            else:
                self.unit_direction_ = Vec2d(dx / self.length_, dy / self.length_)
            self.heading_ = math.atan2(self.unit_direction_.y_, self.unit_direction_.x_)
    
    def start(self):
        return self.start_

    def end(self):
        return self.end_
    
    def length(self):
        return self.length_

    def distance_to(self, point, nearest_pt=None):
        if self.length_ <= kMathEpsilon:
            if nearest_pt is not None:
                nearest_pt.x_ = self.start_.x_
                nearest_pt.y_ = self.start_.y_
            return point.distance_to(self.start_)

        x0 = point.x_ - self.start_.x_
        y0 = point.y_ - self.start_.y_
        proj = x0 * self.unit_direction_.x_ + y0 * self.unit_direction_.y_

        if proj < 0.0:
            if nearest_pt is not None:
                nearest_pt.x_ = self.start_.x_
                nearest_pt.y_ = self.start_.y_
            return math.hypot(x0, y0)

        if proj > self.length_:
            if nearest_pt is not None:
                nearest_pt.x_ = self.end_.x_
                nearest_pt.y_ = self.end_.y_
            return point.distance_to(self.end_)

        if nearest_pt is not None:
            nearest_pt.x_ = self.start_.x_ + self.unit_direction_.x_ * proj
            nearest_pt.y_ = self.start_.y_ + self.unit_direction_.y_ * proj
        return abs(x0 * self.unit_direction_.y_ - y0 * self.unit_direction_.x_)
    
class Box2d:
    def __init__(self, center, length, width, heading):
        self.center_ = center
        self.length_ = length
        self.width_ = width
        self.half_length_ = length / 2.0
        self.half_width_ = width / 2.0
        self.heading_ = heading
        self.cos_heading_ = math.cos(heading)
        self.sin_heading_ = math.sin(heading)
        self.cross_prod_ = cross_prod
        self.corners_ = []
        self.max_x_ = -math.inf
        self.min_x_ = math.inf
        self.max_y_ = -math.inf
        self.min_y_ = math.inf

    def get_all_corners(self):
        return self.corners_

    def max_x(self):
        return self.max_x_

    def min_x(self):
        return self.min_x_

    def max_y(self):
        return self.max_y_

    def min_y(self):
        return self.min_y_

    def is_inside_rectangle(self, point):
        return 0.0 <= point.x() <= self.width_ and 0.0 <= point.y() <= self.length_
    
    def is_point_in(self, point):
        x0 = point.x() - self.center_.x()
        y0 = point.y() - self.center_.y()
        dx = abs(x0 * self.cos_heading_ + y0 * self.sin_heading_)
        dy = abs(-x0 * self.sin_heading_ + y0 * self.cos_heading_)
        return dx <= self.half_length_ + kMathEpsilon and dy <= self.half_width_ + kMathEpsilon

    def has_overlap(self, line_segment):
        if line_segment.length() <= kMathEpsilon:
            return self.is_point_in(line_segment.start())

        if (max(line_segment.start().x(), line_segment.end().x()) < self.min_x() or
            min(line_segment.start().x(), line_segment.end().x()) > self.max_x() or
            max(line_segment.start().y(), line_segment.end().y()) < self.min_y() or
            min(line_segment.start().y(), line_segment.end().y()) > self.max_y()):
            return False

        x_axis = Vec2d(self.sin_heading_, -self.cos_heading_)
        y_axis = Vec2d(self.cos_heading_, self.sin_heading_)

        start_v = line_segment.start() - self.corners_[2]
        start_point = Vec2d(start_v.inner_prod(x_axis), start_v.inner_prod(y_axis))

        if self.is_inside_rectangle(start_point):
            return True

        end_v = line_segment.end() - self.corners_[2]
        end_point = Vec2d(end_v.inner_prod(x_axis), end_v.inner_prod(y_axis))

        if self.is_inside_rectangle(end_point):
            return True

        if start_point.x() < 0.0 and end_point.x() < 0.0:
            return False
        if start_point.y() < 0.0 and end_point.y() < 0.0:
            return False
        if start_point.x() > self.width_ and end_point.x() > self.width_:
            return False
        if start_point.y() > self.length_ and end_point.y() > self.length_:
            return False

        line_direction = line_segment.end() - line_segment.start()
        normal_vec = Vec2d(line_direction.y(), -line_direction.x())
        p1 = self.center_ - line_segment.start()
        diagonal_vec = self.center_ - self.corners_[0]

        project_p1 = abs(p1.inner_prod(normal_vec))
        if abs(diagonal_vec.inner_prod(normal_vec)) >= project_p1:
            return True

        diagonal_vec = self.center_ - self.corners_[1]
        if abs(diagonal_vec.inner_prod(normal_vec)) >= project_p1:
            return True

        return False
    
    def distance_to(self, point):
        x0 = point.x_ - self.center_.x_
        y0 = point.y_ - self.center_.y_
        dx = abs(x0 * self.cos_heading_ + y0 * self.sin_heading_) - self.half_length_
        dy = abs(x0 * self.sin_heading_ - y0 * self.cos_heading_) - self.half_width_

        if dx <= 0.0:
            return max(0.0, dy)
        if dy <= 0.0:
            return dx
        return math.hypot(dx, dy)

    @staticmethod
    def pt_seg_distance(query_x, query_y, start_x, start_y, end_x, end_y, length):
        x0 = query_x - start_x
        y0 = query_y - start_y
        dx = end_x - start_x
        dy = end_y - start_y
        proj = x0 * dx + y0 * dy
        if proj <= 0.0:
            return math.hypot(x0, y0)
        if proj >= length * length:
            return math.hypot(x0 - dx, y0 - dy)
        return abs(x0 * dy - y0 * dx) / length