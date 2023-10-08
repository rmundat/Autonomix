#pragma once

#include <cmath>
#include <vector>

// Splines
std::vector<double> catmullRomSpline(const std::vector<double> &P,
                                     int numPoints) {
  std::vector<double> result;
  double t0, t1, t2, t3, a1, a2, a3, a4;
  for (int i = 1; i < P.size() - 2; i++) {
    for (int j = 0; j < numPoints; j++) {
      double t = (double)j / numPoints;
      t0 = std::pow((1 - t), 3) / 6.0;
      t1 = (3 * t * t * t - 6 * t * t + 4) / 6.0;
      t2 = (-3 * t * t * t + 3 * t * t + 3 * t + 1) / 6.0;
      t3 = t * t * t / 6.0;
      result.push_back(t0 * P[i - 1] + t1 * P[i] + t2 * P[i + 1] +
                       t3 * P[i + 2]);
    }
  }
  return result;
}