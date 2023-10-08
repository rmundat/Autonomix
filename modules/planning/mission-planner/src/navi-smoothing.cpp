#include "common/ad-utils/data-utils.h"
#include "common/ad-utils/map-utils.h"
#include "interface/src/client-consumers.h"
#include "planning/curvature-estimation/src/discrete-curves.h"
#include "planning/mission-planner/src/navi-smoothing.h"

using namespace std;

deque<double> MissionPlanner::GlobalWaypoints::xdq;
deque<double> MissionPlanner::GlobalWaypoints::ydq;

void MissionPlanner::smoothWaypoints() {

    if (wpTransform.x.size() < pathConfig.numWaypoints  || wpTransform.y.size() < pathConfig.numWaypoints) {
        return; 
    }

    const double distToLastWp = euclideanDistance(currentEgoState.x, currentEgoState.y, 
                                                    wpTransform.x.back(), wpTransform.y.back());
    if (distToLastWp < pathConfig.goalDist) {
        // TODO: Check if goal waypoint is in front of the ego vehicle
        currentEgoControl.checkpoint = true;
    }

    if (plotData.xcrs.empty() && plotData.ycrs.empty()) {
        double curveBoundGlobal = 0.5;
        for(size_t i = 1; i < wpTransform.x.size() - 1; i++) {
            plotData.curvature.push_back(computeCurvatureAndSmooth(wpTransform.x, wpTransform.y, i, curveBoundGlobal));
        }

        computeSpline();
    }

    // Get the next waypoint beyond the vehicle wheelbase 
    if (!GlobalWaypoints::xdq.empty() && !GlobalWaypoints::ydq.empty()) {
        int wpNext = nextWaypoint(currentEgoState.x, currentEgoState.y, currentEgoState.psi, 
                                        GlobalWaypoints::xdq, GlobalWaypoints::ydq);
        
        if (wpNext < 0) wpNext = 0;   // Error! Stop!

        for (size_t i = 0; i < wpNext; ++i) {
            GlobalWaypoints::xdq.pop_front();
            GlobalWaypoints::ydq.pop_front();
        }
    }

    static size_t wpIdx = 0;
    double wpDistance = euclideanDistance(currentEgoState.x, currentEgoState.y, plotData.xcrs[wpIdx], plotData.ycrs[wpIdx]);
    // TODO: Use velocity to adjust distance threshold and max(threshold, distance to next waypoint)
    while (wpDistance < pathConfig.distThres  && wpIdx < plotData.xcrs.size()) {
        GlobalWaypoints::xdq.push_back(plotData.xcrs[wpIdx]);
        GlobalWaypoints::ydq.push_back(plotData.ycrs[wpIdx]);
        wpDistance = euclideanDistance(currentEgoState.x, currentEgoState.y, plotData.xcrs[wpIdx], plotData.ycrs[wpIdx]);
        wpIdx++;
    }
}

double MissionPlanner::computeCurvatureAndSmooth(vector<double>& x, vector<double>& y, size_t i, double curveConstraint) {
    double curvature;
    size_t window = 2;
    do {
        curvature = centralDifference(x[i-1], y[i-1], x[i], y[i], x[i+1], y[i+1]);
        if (abs(curvature) > curveConstraint) {
            // Define the range of indices to be smoothed
            size_t start = std::max(i - window, static_cast<size_t>(0));
            size_t end = std::min(i + window + 1, x.size());

            double sumX = 0, sumY = 0;
            for (size_t j = start; j < end; ++j) {
                sumX += x[j];
                sumY += y[j];
            }
            x[i] = sumX / (end - start);
            y[i] = sumY / (end - start);

            for (size_t j = start; j < end; ++j) {
                if (j != i && j > 0 && j < x.size() - 1) {
                    x[j] = (x[j - 1] + x[j + 1]) / 2.0;
                    y[j] = (y[j - 1] + y[j + 1]) / 2.0;
                }
            }
        }
    // TODO: While loop termination based on delta curvature and dynamic curveConstraint
    } while(abs(curvature) > curveConstraint);
    return curvature;
}

void MissionPlanner::computeSpline() {
    wpTransform.x.insert(wpTransform.x.begin(), *wpTransform.x.begin());
    wpTransform.y.insert(wpTransform.y.begin(), *wpTransform.y.begin());
    wpTransform.x.push_back(wpTransform.x.back());
    wpTransform.y.push_back(wpTransform.y.back());
    plotData.xcrs = catmullRomSpline(wpTransform.x, 10);
    plotData.ycrs = catmullRomSpline(wpTransform.y, 10);
}

int MissionPlanner::nextWaypoint(double x, double y, double psi, const deque<double> &wpx, const deque<double> &wpy) {
    int wpClosest = -1;
    double distClosest = numeric_limits<double>::max();
    double minDotProduct = 0.0; 

    for (int i = 0; i < wpx.size(); ++i) {
        double dx = wpx[i] - x;
        double dy = wpy[i] - y;
        double dist = sqrt(dx * dx + dy * dy);

        // Calculate the dot product to determine if the waypoint is in front of the vehicle
        double dotProduct = dx * cos(psi) + dy * sin(psi);

        if (dist < distClosest && dotProduct > minDotProduct) {
            distClosest = dist;
            wpClosest = i;
        }
    }
    return wpClosest;
}