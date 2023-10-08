#pragma once

#include <cmath>
#include <vector>


// Simple moving average
std::vector<double> movingAverage(const std::vector<double> &data, int window) {
  std::vector<double> result;
  for (int i = 0; i < data.size(); ++i) {
    double total = 0.0;
    int count = 0;
    for (int j = std::max(0, i - window);
         j <= std::min(static_cast<int>(data.size()) - 1, i + window); ++j) {
      total += data[j];
      ++count;
    }
    result.push_back(total / count);
  }
  return result;
}

// Gaussian window filter
std::vector<double> gaussianSmooth(const std::vector<double> &data,
                                   int windowSize, double sigma) {
  // Calculate the weights for the Gaussian kernel
  std::vector<double> weights;
  double weightSum = 0.0;
  for (int i = -windowSize; i <= windowSize; ++i) {
    double weight = exp(-(i * i) / (2 * sigma * sigma));
    weights.push_back(weight);
    weightSum += weight;
  }

  // Normalize the weights
  for (double &weight : weights) {
    weight /= weightSum;
  }

  // Apply the Gaussian kernel
  std::vector<double> result(data.size());
  for (int i = 0; i < data.size(); ++i) {
    double total = 0.0;
    for (int j = 0; j < weights.size(); ++j) {
      int dataIndex = i - windowSize + j;
      if (dataIndex >= 0 && dataIndex < data.size()) {
        total += data[dataIndex] * weights[j];
      }
    }
    result[i] = total;
  }

  return result;
}
