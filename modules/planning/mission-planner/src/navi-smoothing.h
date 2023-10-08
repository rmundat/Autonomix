#pragma once

#include <vector>

struct PathPlannerConfig {
    double startX;
    double startY;
    double direction;
    double distThres;
    double goalDist;
    double wpSlopeThres;
    int numWaypoints;
};

class MissionPlanner {
public:
    struct GlobalWaypoints {
        static std::deque<double> xdq, ydq;
    };

    MissionPlanner()
        :pathConfig({
            .startX = 0.0,
            .startY = 0.0,
            .direction = 0.0, 
            .distThres = 10.0,
            .goalDist = 2.0,
            .wpSlopeThres = 10000.0,
            .numWaypoints = 50}) {}
    
    virtual ~MissionPlanner() {}
    
    void smoothWaypoints();
    static std::deque<double>& getXdq() { return GlobalWaypoints::xdq; }
    static std::deque<double>& getYdq() { return GlobalWaypoints::ydq; }

private:
    PathPlannerConfig pathConfig;
    double computeCurvatureAndSmooth(std::vector<double>& x, std::vector<double>& y, size_t i, double curveConstraint);
    void computeSpline();
    int nextWaypoint(double x, double y, double psi, const std::deque<double> &wpx, const std::deque<double> &wpy);
};