#pragma once

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <casadi/casadi.hpp>

// MPC horizon
#define NUMBER_OF_STEPS 10

// Set desired cross track, heading errors and reference velocity (kph)
#define REF_CTE 0
#define REF_EPSI 0
#define REF_V 20          

// Weights for the cost function
#define W_CTE 0.3       // Cross-track error weight
#define W_EPSI 0.5      // Heading error weight
#define W_V 1.0         // Velocity error weight
#define W_DELTA 1.0     // Steering actuator weight
#define W_A 1.0         // Throttle actuator weight
#define W_DDELTA 1.0    // Change in steering actuator weight
#define W_DA 1.0        // Change in throttle actuator weight

// Set lower and upper limits for variables.
#define DED25RAD 0.436332 // 25 deg in rad, used as delta bound
#define MAXTHR 1.0        // Maximal a value
#define BOUND 1.0e3       // Bound value for other variables


struct ControlInput {
    double deltaF;  // Front wheel steering angle (rad)
    double Fx;      // Longitudinal force (N)
};

// Structure to hold the control inputs over the prediction horizon
struct HorizonControlInput {
    CppAD::vector<CppAD::AD<double>> deltaF;
    CppAD::vector<CppAD::AD<double>> cmdVx;
};

// MPC class definition
class MPC {
public:
    MPC();
    virtual ~MPC();

    std::vector<double> solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);
    std::vector<double> mpcX;
    std::vector<double> mpcY;

    // static Eigen::VectorXd polyfit(const std::vector<double>& xvals, const std::vector<double>& yvals, int order);
    // static double polyeval(Eigen::VectorXd coeffs, double x);
};

class FGeval {
public:

    // Fitted polynomial coefficients
    Eigen::VectorXd coeffs;
    FGeval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

    typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    void operator()(ADvector& fg, const ADvector& vars);

private:
    std::vector<double> _coeffs;
    double _targetVelocity;
};