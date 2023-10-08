#include "common/ad-utils/data-utils.h"
#include <carla/geom/Location.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace planning {

struct GraphNode {
  int id;
  carla::geom::Location loc;
};

struct GraphEdge {
  int src;
  int dst;
};

// GCC-specific visibility attribute
class __attribute__((visibility("default"))) FreeSpaceNode {
public:
  FreeSpaceNode();
  ~FreeSpaceNode() = default;

  void setWorld(const std::shared_ptr<py::object> &world);
  // std::shared_ptr<py::object> getWorld() const { return world_; }
  // FreeSpaceNode(const std::shared_ptr<py::object>& world) : world_(world) {}

  void processWorld();

private:
  std::shared_ptr<py::object> world_;
  std::vector<std::map<std::string, py::object>>
  buildTopology(const py::object &roadTopology);

  const double samplingResolution = 1.0;
  std::vector<std::map<std::string, double>> bounding_boxes;
};

} // namespace planning