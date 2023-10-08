#include "common/constants.h"
#include "common/ad-utils/data-utils.h"
#include "interface/src/client-consumers.h"
#include "planning/mission-planner/src/navi-smoothing.h"
#include "setup-mpc.h"
#include "motion-control.h"

using namespace std;
using CppAD::AD;

const double dt = DT_05s;
constexpr double MAX_THROTTLE = 0.8;
constexpr double MAX_BRAKE = 0.5;
constexpr double STEER_LIMIT = 25;
constexpr int POLY_ORDER = 3;

// Single track vehicle dynamics equations -> update to dual track
Eigen::VectorXd DualTrackVehicle::egoModel(double cte, double epsi) {

    // Approximations, to be replaced with accurate estimations 
    const double deltaF = currentEgoControl.steer * deg2rad(STEER_LIMIT);
    const double a = currentEgoControl.throttle;

    // https://projectswiki.eleceng.adelaide.edu.au/projects/index.php/File:SSM.PNG
    float a11 = -(safeDivide((_params.Cf + _params.Cr), (_params.m * currentEgoState.vx)));
    float a12 = -(safeDivide(currentEgoState.vx - (_params.a * _params.Cf - _params.b * _params.Cr), (_params.m * currentEgoState.vx)));
    float a21 = -(safeDivide((_params.a * _params.Cf - _params.b * _params.Cr), _params.Iz));
    float a22 = -(safeDivide((_params.a * _params.a * _params.Cf + _params.b * _params.b * _params.Cr), (_params.Iz * currentEgoState.vx)));

    float B1 = safeDivide(_params.Cf, _params.m);
    float B2 = safeDivide(_params.a * _params.Cf, _params.Iz);

    // State variables
    Eigen::VectorXd x(6);
    x << 0.0, 0.0, 0.0, currentEgoState.vx, cte, epsi;

    // State update
    const double psiDot = (currentEgoState.vx / _params.Lf) * tan(deltaF); // Yaw rate
    Eigen::VectorXd xdot(6);
    xdot << currentEgoState.vx * cos(currentEgoState.psi),
            currentEgoState.vx * sin(currentEgoState.psi),
            psiDot,
            a11 * currentEgoState.vx + a12 * psiDot + B1 * deltaF,
            currentEgoState.vx * sin(epsi),
            psiDot;

    Eigen::VectorXd _state(6);
    _state = x + xdot * dt;

    return _state;
}

double polyeval(Eigen::VectorXd coeffs, double x) {
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); i++) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}

Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) {
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);

    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }

    for (int j = 0; j < xvals.size(); j++) {
        for (int i = 0; i < order; i++) {
            A(j, i + 1) = A(j, i) * xvals[j];
        }
    }

    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}

void motionControl() {

    shared_ptr<MissionPlanner> missionPlanner = make_shared<MissionPlanner>();

    missionPlanner->smoothWaypoints();
    auto& xdq = missionPlanner->getXdq();
    auto& ydq = missionPlanner->getYdq();

    if (xdq.empty() || ydq.empty()) return;

    // Waypoints to vehicle coordinates transform:
    vector<double> wpxLocal; // vector<double>::reserve
    vector<double> wpyLocal;
    wpxLocal.resize(xdq.size());
    wpyLocal.resize(ydq.size());

    for (size_t wpIdx = 0; wpIdx < xdq.size(); wpIdx++) {
        double dx = xdq[wpIdx] - currentEgoState.x;
        double dy = ydq[wpIdx] - currentEgoState.y;
        wpxLocal[wpIdx] = dx * cos(currentEgoState.psi) + dy * sin(currentEgoState.psi);
        wpyLocal[wpIdx] = -dx * sin(currentEgoState.psi) + dy * cos(currentEgoState.psi);
    }

    // Convert the STL vectors into Eigen::VectorXd
    Eigen::VectorXd xvals = Eigen::VectorXd::Map(wpxLocal.data(), wpxLocal.size());
    Eigen::VectorXd yvals = Eigen::VectorXd::Map(wpyLocal.data(), wpyLocal.size());

    // Fit polynomial to the transformed waypoints to obtain reference trajectory from vehicle perspective
    Eigen::VectorXd coeffs = polyfit(xvals, yvals, POLY_ORDER);
    // x, y = 0.0 in local frame | cte = coeffs[0] + coeffs[1] * x0 + coeffs[2] * pow(x0, 2) + coeffs[3] * pow(x0, 3);
    double cte = polyeval(coeffs, 0);

    // Initial heading error. The desired orientation is tangent to the reference trajectory at x = 0
    double df0 = coeffs[1]; 
    double psides0 = atan(df0); 
    double epsi = -psides0;  

    if (!isfinite(coeffs[0]) || !isfinite(coeffs[1]) || !isfinite(coeffs[2]) || !isfinite(coeffs[3])) {
        // Probably pop and continue
        cout << "Wapoint erraric!" << endl;
        xdq.pop_back();
        ydq.pop_back();
        return;
    }    

    DualTrackVehicle egoVehicle(egoParams);
    Eigen::VectorXd state = egoVehicle.egoModel(cte, epsi);

    MPC mpc;
    vector<double> mpcout = mpc.solve(state, coeffs);
    // scoped_lock<mutex> lock(currentEgoControlMutex); -> use unique with condition_variable
    currentEgoControl.x = mpcout[0];
    currentEgoControl.y = mpcout[1];
    currentEgoControl.psi = mpcout[2];
    currentEgoControl.v = mpcout[3];
    currentEgoControl.cte = mpcout[4];
    currentEgoControl.epsi = mpcout[5];
    currentEgoControl.steer = (mpcout[6]/ deg2rad(25)); // convert to [-1..1]
    currentEgoControl.mpcost = mpcout[8]; 
    if (mpcout[7] > 0) {
        currentEgoControl.brake = 0.0;
        currentEgoControl.throttle = min(clamp(mpcout[7], 0.0, 1.0), MAX_THROTTLE);
    } else {
        currentEgoControl.throttle = 0.0;
        currentEgoControl.brake = min(abs(mpcout[7]), MAX_BRAKE);
    } 
}