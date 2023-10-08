#pragma once

#include <set>
#include <deque>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <cppad/cppad.hpp>

class DualTrackVehicle {
public:
    DualTrackVehicle(const EgoParameters& params) : _params(params) {}

    Eigen::VectorXd egoModel(double cte, double epsi);

private:
    EgoParameters _params;
};

double polyeval(Eigen::VectorXd coeffs, double x);
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);
void motionControl();