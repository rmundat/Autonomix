#include "planning/route-planner/src/route-handler.h"

// // Define origin and destination
// carla::geom::Location origin(13.5, -13.7, 0.0);
// carla::geom::Location destination(180, 240, 0.5);

namespace planning {
planning::FreeSpaceNode::FreeSpaceNode() {}

void FreeSpaceNode::setWorld(const std::shared_ptr<py::object> &world) {
  world_ = world;
}

std::vector<std::map<std::string, py::object>>
FreeSpaceNode::buildTopology(const py::object &roadTopology) {
  std::vector<std::map<std::string, py::object>> topology;

  for (const auto &segment : roadTopology) {
    py::object wp1 = segment.attr("__getitem__")(0);
    py::object wp2 = segment.attr("__getitem__")(1);

    auto transform1 = wp1.attr("transform");
    auto transform2 = wp2.attr("transform");

    auto l1 = transform1.attr("location");
    auto l2 = transform2.attr("location");

    double x1 = round(l1.attr("x").cast<double>());
    double y1 = round(l1.attr("y").cast<double>());
    double z1 = round(l1.attr("z").cast<double>());
    double x2 = round(l2.attr("x").cast<double>());
    double y2 = round(l2.attr("y").cast<double>());
    double z2 = round(l2.attr("z").cast<double>());

    double dist = euclideanDistance(x1, y1, x2, y2);

    // Compute the normalized direction vector
    double dx = x2 - x1;
    double dy = y2 - y1;
    double magnitude = sqrt(dx * dx + dy * dy);
    dx /= magnitude;
    dy /= magnitude;

    // Compute the perpendicular vector
    double perp_dx = -dy;
    double perp_dy = dx;

    // Getting the lane width for both waypoints
    double lane_width1 = wp1.attr("lane_width").cast<double>();
    double lane_width2 = wp2.attr("lane_width").cast<double>();

    // lane boundaries for the waypoints using the perpendicular vector
    double x1_left_boundary = x1 + perp_dx * (lane_width1 / 2.0);
    double y1_left_boundary = y1 + perp_dy * (lane_width1 / 2.0);
    double x1_right_boundary = x1 - perp_dx * (lane_width1 / 2.0);
    double y1_right_boundary = y1 - perp_dy * (lane_width1 / 2.0);

    double x2_left_boundary = x2 + perp_dx * (lane_width2 / 2.0);
    double y2_left_boundary = y2 + perp_dy * (lane_width2 / 2.0);
    double x2_right_boundary = x2 - perp_dx * (lane_width2 / 2.0);
    double y2_right_boundary = y2 - perp_dy * (lane_width2 / 2.0);

    // Compute bounding box for the segment
    double x_min = std::min({x1_left_boundary, x1_right_boundary,
                             x2_left_boundary, x2_right_boundary});
    double x_max = std::max({x1_left_boundary, x1_right_boundary,
                             x2_left_boundary, x2_right_boundary});
    double y_min = std::min({y1_left_boundary, y1_right_boundary,
                             y2_left_boundary, y2_right_boundary});
    double y_max = std::max({y1_left_boundary, y1_right_boundary,
                             y2_left_boundary, y2_right_boundary});

    // Store the bounding box for this segment
    std::map<std::string, double> bbox = {
        {"x_min", x_min}, {"x_max", x_max}, {"y_min", y_min}, {"y_max", y_max}};
    bounding_boxes.push_back(bbox);
  }
  return topology;
}

void FreeSpaceNode::processWorld() {
  // Retrieve the map and road topology
  auto simMap = world_->attr("get_map")();
  auto roadTopology = simMap.attr("get_topology")();

  auto topology = buildTopology(roadTopology);

  // Retrieve the vehicle and its transform
  auto vehicle = world_->attr("get_actors")().attr("filter")("vehicle.*")[0];
  auto vehicleTransform = vehicle.attr("get_transform")();

  // Extract the vehicle's start position and rotation
  double startX = vehicleTransform.attr("location").attr("x").cast<double>();
  double startY = vehicleTransform.attr("location").attr("y").cast<double>();
  double startPhi =
      vehicleTransform.attr("rotation").attr("yaw").cast<double>();
}

} // namespace planning