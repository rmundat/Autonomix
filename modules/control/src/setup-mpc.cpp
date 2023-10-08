#include "common/constants.h"
#include "interface/src/client-consumers.h"
#include "motion-control.h"
#include "setup-mpc.h"

using namespace std;
using CppAD::AD;

// Prediction horizon and discretization
const size_t N = NUMBER_OF_STEPS; // Number of time steps in the prediction horizon
const double dt = DT_05s;         // Time step size (discretization)

// The target speed in meters per second
const double refV = REF_V * 0.277778; 

// The solver takes all the state variables and actuator variables in a singular vector
size_t xStart = 0;
size_t yStart = xStart + N;
size_t psiStart = yStart + N;
size_t vStart = psiStart + N;
size_t cteStart = vStart + N;
size_t epsiStart = cteStart + N;
size_t deltaStart = epsiStart + N;
size_t aStart = deltaStart + N - 1;

void FGeval::operator()(ADvector& fg, const ADvector& vars) {
    fg[0] = 0;

    // Minimize state costs
    for (int i = 0; i < N; i++) {
      fg[0] += W_CTE * CppAD::pow(vars[cteStart + i] - REF_CTE, 2);
      fg[0] += W_EPSI * CppAD::pow(vars[epsiStart + i] - REF_EPSI, 2);
      fg[0] += W_V * CppAD::pow(vars[vStart + i] - refV, 2);
    }

    // Minimize the use of actuators
    for (int i = 0; i < N - 1; i++) {
      fg[0] += W_DELTA * CppAD::pow(vars[deltaStart + i], 2);
      fg[0] += W_A * CppAD::pow(vars[aStart + i], 2);
    }

    // Minimize the value gap between sequential actuations
    for (int i = 0; i < N - 2; i++) {
      fg[0] += W_DDELTA * CppAD::pow(vars[deltaStart + i + 1] - vars[deltaStart + i], 2);
      fg[0] += W_DA * CppAD::pow(vars[aStart + i + 1] - vars[aStart + i], 2);
    }

    // Set up the initial constraints with the shift
    // Add 1 to each of the starting indices of 'fg' since cost is at index 0 of `fg`.
    fg[1 + xStart] = vars[xStart];
    fg[1 + yStart] = vars[yStart];
    fg[1 + psiStart] = vars[psiStart];
    fg[1 + vStart] = vars[vStart];
    fg[1 + cteStart] = vars[cteStart];
    fg[1 + epsiStart] = vars[epsiStart];

    for (int t = 0; t < N - 1; t++) {

        // State at time t + 1
        AD<double> x1 = vars[xStart + t + 1];
        AD<double> y1 = vars[yStart + t + 1];
        AD<double> psi1 = vars[psiStart + t + 1];
        AD<double> vx1 = vars[vStart + t + 1];
        AD<double> cte1 = vars[cteStart + t + 1];
        AD<double> epsi1 = vars[epsiStart + t + 1];

        // State at time t
        AD<double> x0 = vars[xStart + t];
        AD<double> y0 = vars[yStart + t];
        AD<double> psi0 = vars[psiStart + t];
        AD<double> vx0 = vars[vStart + t];
        AD<double> cte0 = vars[cteStart + t];
        AD<double> epsi0 = vars[epsiStart + t];

        // Actuation at time t
        AD<double> deltaF0 = vars[deltaStart + t];
        AD<double> a0 = vars[aStart + t];

        // Reference trajectory evaluated at x_t
        AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * CppAD::pow(x0, 2) + coeffs[3] * CppAD::pow(x0, 3);
        // Derivative of the reference trajectory evaluated at x_t
        AD<double> df0 = coeffs[1] + 2 * coeffs[2] * x0 + 3 * coeffs[3] * CppAD::pow(x0, 2);
        AD<double> psides0 = CppAD::atan(df0);

        // State update equations -> simplified to kinamatic model for testing 
        fg[2 + xStart + t] = x1 - (x0 + vx0 * cos(psi0) * dt);
        fg[2 + yStart + t] = y1 - (y0 + vx0 * sin(psi0) * dt);
        fg[2 + psiStart + t] = psi1 - (psi0 + vx0 * deltaF0 / egoParams.Lf * dt);
        fg[2 + vStart + t] = vx1 - (vx0 + a0 * dt);
        fg[2 + cteStart + t] = cte1 - ((f0 - y0) + (vx0 * sin(epsi0) * dt)); // Using (f0 - y0) for the cte is a pretty terrible approximation, especially for sharper turns
        fg[2 + epsiStart + t] = epsi1 - ((psi0 - psides0) + vx0 * deltaF0 / egoParams.Lf * dt);
    }
}

// Minimize total cost using an optimization solver like cppad::ipopt
// ...
// MPC class definition implementation
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {

    // size_t i;
    typedef CPPAD_TESTVECTOR(double) Dvector;

    double x = state[0];
    double y = state[1];
    double psi = state[2];
    double v = state[3];
    double cte = state[4];
    double epsi = state[5];

    // Set the number of model variables (state is a 4 element vector, the actuators is a 2 and 20 timesteps)
    // N timesteps == N - 1 actuations
    size_t n_vars = N * 6 + (N - 1) * 2;
    // Number of constraints
    size_t n_constraints = N * 6;

    // Initial value of the independent variables.
    // Should be 0 besides initial state.
    Dvector vars(n_vars);
    for (int i = 0; i < n_vars; i++) {
        vars[i] = 0;
    }

    // Set the initial variable values
    vars[xStart] = x;
    vars[yStart] = y;
    vars[psiStart] = psi;
    vars[vStart] = v;
    vars[cteStart] = cte;
    vars[epsiStart] = epsi;

    Dvector vars_lowerbound(n_vars);
    Dvector vars_upperbound(n_vars);

    // Set all non-actuators upper and lower limits
    // to the max negative and positive values.
    for (int i = 0; i < deltaStart; i++) {
        vars_lowerbound[i] = -1.0e19;
        vars_upperbound[i] = 1.0e19;
    }

    // The upper and lower limits of delta are set to -25 and 25
    // degrees (values in radians).
    for (int i = deltaStart; i < aStart; i++) {
        vars_lowerbound[i] = -0.436332;
        vars_upperbound[i] = 0.436332;
    }

    // Acceleration/deceleration upper and lower limits.
    for (int i = aStart; i < n_vars; i++) {
        vars_lowerbound[i] = -1.0;
        vars_upperbound[i] = 1.0;
    }

    // Lower and upper limits for the constraints
    // Should be 0 besides initial state.
    Dvector constraints_lowerbound(n_constraints);
    Dvector constraints_upperbound(n_constraints);
    for (int i = 0; i < n_constraints; i++) {
        constraints_lowerbound[i] = 0;
        constraints_upperbound[i] = 0;
    }
    constraints_lowerbound[xStart] = x;
    constraints_lowerbound[yStart] = y;
    constraints_lowerbound[psiStart] = psi;
    constraints_lowerbound[vStart] = v;
    constraints_lowerbound[cteStart] = cte;
    constraints_lowerbound[epsiStart] = epsi;

    constraints_upperbound[xStart] = x;
    constraints_upperbound[yStart] = y;
    constraints_upperbound[psiStart] = psi;
    constraints_upperbound[vStart] = v;
    constraints_upperbound[cteStart] = cte;
    constraints_upperbound[epsiStart] = epsi;

    // Object that computes objective and constraints
    FGeval fgeval(coeffs);

    // Options for IPOPT solver
    std::string options;
    // Uncomment this if you'd like more print information
    options += "Integer print_level  0\n";
    // NOTE: Setting sparse to true allows the solver to take advantage
    // of sparse routines, this makes the computation MUCH FASTER. If you
    // can uncomment 1 of these and see if it makes a difference or not but
    // if you uncomment both the computation time should go up in orders of
    // magnitude.
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    // NOTE: Currently the solver has a maximum time limit of 0.5 seconds
    options += "Numeric max_cpu_time          0.5\n";

    // Place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // Solve the problem
    CppAD::ipopt::solve<Dvector, FGeval>(options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
        constraints_upperbound, fgeval, solution);

    // Check some of the solution values
    bool ok = true;
    ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

    // Convert the solution to an STL vector
    auto cost = solution.obj_value;
    return {solution.x[xStart + 1],   solution.x[yStart + 1],
            solution.x[psiStart + 1], solution.x[vStart + 1],
            solution.x[cteStart + 1], solution.x[epsiStart + 1],
            solution.x[deltaStart],   solution.x[aStart], cost};

};
