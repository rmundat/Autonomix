#include "common/ad-utils/matplotlibcpp.h"
#include "interface/src/client-consumers.h"
#include "planning/route-planner/src/route-handler.h"

namespace py = pybind11;
namespace plt = matplotlibcpp;

int main() {
  try {
    // PS: Initialize the Python interpreter
    py::scoped_interpreter guard{};

    // Convert py::str to std::string before printing
    py::module sys = py::module::import("sys");
    std::string pythonExecutable = py::str(sys.attr("executable"));

    // Import the CarlaSim Python module
    py::module carlaSim = py::module::import("simulation.src.carla-wrapper");

    // Create an instance of the CarlaSim class
    py::object carlaSimInstance = carlaSim.attr("CarlaSim")();

    // Retrieve Carla world
    py::object world = carlaSimInstance.attr("getWorld")();
    std::shared_ptr<py::object> carlaWorldObj =
        std::make_shared<py::object>(world);

    // Set and process world object
    planning::FreeSpaceNode freeSpaceNode;
    freeSpaceNode.setWorld(carlaWorldObj);
    freeSpaceNode.processWorld();

    // Call the runCarla function with the thread safe callback function
    carlaSimInstance.attr("runCarla")(
        py::cpp_function([&]() {
          std::scoped_lock<std::mutex> lock(currentEgoControlMutex);
          return currentEgoControl.steer;
        }),
        py::cpp_function([&]() {
          std::scoped_lock<std::mutex> lock(currentEgoControlMutex);
          return currentEgoControl.throttle;
        }),
        py::cpp_function([&]() {
          std::scoped_lock<std::mutex> lock(currentEgoControlMutex);
          return currentEgoControl.brake;
        }),
        py::cpp_function([&]() {
          std::scoped_lock<std::mutex> lock(currentEgoControlMutex);
          return currentEgoControl.checkpoint;
        }),
        py::cpp_function(
            [&](py::dict egoVehicleInformation, py::list waypointsList) {
              processVehicleInformation(egoVehicleInformation, waypointsList);
            }));

    plt::figure();
    plt::title("waypoint");
    plt::named_plot("ego(loc)", plotData.xEgo, plotData.yEgo);
    plt::named_plot("world(wp)", wpTransform.x, wpTransform.y, "bo");
    plt::named_plot("interpolated(wp)", plotData.xcrs, plotData.ycrs, "-go");
    plt::legend();

    plt::figure();
    plt::subplot(2, 1, 1);
    plt::named_plot("CT Error", plotData.cte);
    plt::named_plot("Heading Error", plotData.epsi);
    plt::legend();
    plt::subplot(2, 1, 2);
    plt::named_plot("Steer", plotData.steer);
    plt::named_plot("Heading", plotData.yaw);
    plt::legend();

    plt::figure();
    plt::subplot(2, 1, 1);
    plt::named_plot("Ref Velocity", plotData.vError);
    plt::named_plot("Velocity", plotData.vx);
    plt::legend();
    plt::subplot(2, 1, 2);
    plt::named_plot("Throttle", plotData.app);
    plt::named_plot("Brake", plotData.brake);
    plt::legend();

    plt::figure();
    plt::subplot(1, 1, 1);
    plt::named_plot("MPC Cost", plotData.mpcost);
    plt::legend();

    plt::figure();
    plt::title("Curvature Overlay");
    plt::named_plot("interpolated(wp)", plotData.xcrs, plotData.ycrs, "-bo");
    plt::named_plot("ego(loc)", plotData.xEgo, plotData.yEgo, "-g");
    for (size_t i = 0; i < plotData.curvature.size(); ++i) {
      std::string label = std::to_string(plotData.curvature[i]);
      if (abs(plotData.curvature[i]) > 0.5) {
        plt::plot(std::vector<double>{plotData.xcrs[i]},
                  std::vector<double>{plotData.ycrs[i]}, "ro");
        plt::annotate(label, plotData.xcrs[i], plotData.ycrs[i]);
      }
    }
    plt::legend();

    plt::show();
  } catch (const py::error_already_set &e) {
    std::cerr << "Python error: " << e.what() << std::endl;
    std::cerr << "Python traceback: " << e.trace() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}