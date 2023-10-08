#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


// For converting back and forth between radians and degrees
constexpr double pi() { return std::acos(-1); }
inline double deg2rad(double x) { return x * pi() / 180; }
inline double rad2deg(double x) { return x * 180 / pi(); }

// Divide by zero protection
template <typename T1, typename T2>
inline auto safeDivide(T1 numerator, T2 denominator, T1 defaultInt = 0,
                       T1 defaultFloat = 0.0)
    -> decltype(numerator / denominator) {
  static_assert(std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
                "safeDivide requires numeric types");

  if (std::is_integral<T1>::value && std::is_integral<T2>::value) {
    if (denominator == 0) {
      return defaultInt;
    }
  } else {
    if (denominator < std::numeric_limits<T2>::epsilon()) {
      return defaultFloat;
    }
  }
  return numerator / denominator;
}

// Gradient between two points
inline double pointSlope(double x1, double y1, double x2, double y2) {
  return safeDivide(y2 - y1, x2 - x1);
}

// Euclidean distance between two point
inline double euclideanDistance(double x1, double y1, double x2, double y2) {
  return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Rate limit ascend/descend
inline double rateLimiter(double target, double initial, double increment) {
  if (target > initial + increment) {
    return initial + increment;
  } else if (target < initial - increment) {
    return initial - increment;
  } else {
    return target;
  }
}

// Real Time Rate Limiter
class RateLimiterRT {
public:
  RateLimiterRT(double iv)
      : value(iv), lastUpdate(std::chrono::steady_clock::now()) {}

  double update(double target, double stepSize, double dt) {
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - lastUpdate;
    if (value < target) {
      value += stepSize * elapsed.count();
      if (value > target)
        value = target;
    } else if (value > target) {
      value -= stepSize * elapsed.count();
      if (value < target)
        value = target;
    }
    lastUpdate = now;
    return value;
  }

  double get() const { return value; }

private:
  double value;
  std::chrono::time_point<std::chrono::steady_clock> lastUpdate;
};

// 2D Look Up Table
class LookUpTable {
private:
  std::vector<double> xs;
  std::vector<double> ys;
  std::vector<std::vector<double>> values;

public:
  LookUpTable(const std::vector<double> &xs, const std::vector<double> &ys,
              const std::vector<std::vector<double>> &values)
      : xs(xs), ys(ys), values(values) {}

  double lookup2d(double x, double y) const {
    if (xs.empty() || ys.empty() || values.empty()) {
      return 0.0;
    }

    auto xit = std::lower_bound(xs.begin(), xs.end(), x);
    auto yit = std::lower_bound(ys.begin(), ys.end(), y);

    if (xit == xs.end()) {
      xit = xs.end() - 1;
    }
    if (yit == ys.end()) {
      yit = ys.end() - 1;
    }
    if (x < xs[0]) {
      xit = xs.begin();
    }
    if (y < ys[0]) {
      yit = ys.begin();
    }

    auto i = xit - xs.begin();
    auto j = yit - ys.begin();

    double x1 = xs[i];
    double x2 = xs[i + 1];
    double y1 = ys[j];
    double y2 = ys[j + 1];

    return (values[i][j] * (x2 - x) * (y2 - y) +
            values[i + 1][j] * (x - x1) * (y2 - y) +
            values[i][j + 1] * (x2 - x) * (y - y1) +
            values[i + 1][j + 1] * (x - x1) * (y - y1)) /
           ((x2 - x1) * (y2 - y1));
  }
};
