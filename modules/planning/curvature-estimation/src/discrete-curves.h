#pragma once

#include <cmath>
#include "common/ad-utils/data-utils.h"

double centralDifference(double x1, double y1, double x2, double y2, double x3, double y3) {
    // Compute path segments
    double dx1 = x2 - x1;
    double dy1 = y2 - y1;
    double dx2 = x3 - x2;
    double dy2 = y3 - y2;

    double slope1 = safeDivide(dy1, dx1);
    double slope2 = safeDivide(dy2, dx2);

    // Compute the rate of change of the slope 
    double dmdx = safeDivide((slope2 - slope1), ((dx2 + dx1) / 2));

    // Compute curvature K at the point (x2, y2)
    double curvature = safeDivide(dmdx, pow(1 + pow((slope1 + slope2) / 2, 2), 1.5));

    return curvature;
}

double discreteCurvature(double x1, double y1, double x2, double y2, double x3, double y3) {
    // Compute side lengths
    double a = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
    double b = std::sqrt(std::pow(x3 - x2, 2) + std::pow(y3 - y2, 2));
    double c = std::sqrt(std::pow(x3 - x1, 2) + std::pow(y3 - y1, 2));

    // Compute semiperimeter
    double s = (a + b + c) / 2;

    // Compute the area of the triangle
    double area = std::sqrt(s * (s - a) * (s - b) * (s - c));

    // Compute radius of circumscribed circle and protect for collinear points
    double R = safeDivide((a * b * c), (4 * area));

    // Compute curvature
    double curvature = safeDivide(1, R);

    return curvature;
}
