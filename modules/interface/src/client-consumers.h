#pragma once

#define _USE_MATH_DEFINES

#include <iostream>
#include <utility>
#include <limits>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <cmath>
#include <memory>
#include <mutex>
#include <functional>             // for std::function
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <carla/geom/Location.h>
#include <carla/geom/Rotation.h>
#include <carla/geom/Vector3D.h>

struct CurrentEgoControl {
    double steer;
    double throttle;
    double brake;
    double x;
    double y;
    double psi;
    double v;
    double cte;
    double epsi;
    double mpcost;
    bool checkpoint;
};

struct CurrentEgoState {
    double x;         // x position (m)
    double y;         // y position (m)
    double z;         // y position (m)
    double pitch;
    double yaw;       
    double roll;
    double vx;        // Longitudinal velocity (m/s)
    double vy;        // Lateral velocity (m/s)
    double rotationX; // rad/s
    double rotationY; // rad/s
    double rotationZ; // rad/s
    double psi;       // Yaw angle (rad)
    double r;         // Yaw Rate
    double steer;
    double throttle;
    double brake;
    struct ForwardVectorGlobal {
        double x, y, z;
        ForwardVectorGlobal() : x(0.0), y(0.0), z(0.0) {}
    } forwardVectorGlobal; 
};

struct EgoParameters {
    double a;  // Distance from CG to front axle (m)
    double b;  // Distance from CG to rear axle (m)
    double m;  // Vehicle mass (kg)
    double Iz; // Yaw moment of inertia (kg*m^2)
    double Cf; // Cornering stiffness of the front tires (N/rad)
    double Cr; // Cornering stiffness of the rear tires (N/rad)
    double Lf; // Distance from the vehicle center of mass to the front axle
    double Lr; // Distance from the vehicle center of mass to the rear axle
};

struct WaypointTransform {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
};

struct EgoAttributes {
    double mass;
    double moi;  // Moment of intertia
    double cd;   // Drag coefficient

    // Center of mass offset in x, y and z
    struct CenterOfMass {
        double x, y, z;
    } centerOfMass; 

    struct WheelPhysics {
        std::vector<double> frontLeft;
        std::vector<double> frontRight;
        std::vector<double> rearLeft;
        std::vector<double> rearRight;
    } wheelPhysics;

    struct WheelPosition {
        double x, y, z;
    } wheelPosition; 
};

struct PlotData {
    std::vector<double> xEgo;
    std::vector<double> yEgo;
    std::vector<double> xcrs;
    std::vector<double> ycrs;
    std::vector<double> xInterp;
    std::vector<double> yInterp;
    std::vector<double> yaw;
    std::vector<double> cte;
    std::vector<double> epsi;
    std::vector<double> steer;
    std::vector<double> app;
    std::vector<double> brake;
    std::vector<double> vx;
    std::vector<double> vError;
    std::vector<double> wpcount;
    std::vector<double> mpcost;
    std::vector<double> curvature;
};

extern CurrentEgoState currentEgoState;
extern CurrentEgoControl currentEgoControl;
extern EgoParameters egoParams;
extern WaypointTransform wpTransform;
extern PlotData plotData;
extern std::vector<std::vector<double>> waypoints;
extern std::mutex currentEgoControlMutex;

void processVehicleInformation(pybind11::dict egoVehicleInformation, pybind11::list waypointsList);