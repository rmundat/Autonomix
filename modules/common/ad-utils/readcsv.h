#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


struct CSVData {
  std::vector<double> xv;
  std::vector<double> yv;
  std::vector<std::vector<double>> zv;
};

CSVData readCSV(const std::string &filePath) {
  CSVData data;
  std::ifstream file(filePath);

  if (file.is_open()) {
    std::string line;
    bool firstRow = true;

    while (std::getline(file, line)) {
      std::vector<double> row;
      std::istringstream ss(line);
      std::string val;

      while (std::getline(ss, val, ',')) {
        row.push_back(std::stod(val));
      }

      if (firstRow) {
        data.xv = row;
        firstRow = false;
      } else {
        data.yv.push_back(row[0]);
        row.erase(row.begin());
        data.zv.push_back(row);
      }
    }
  } else {
    std::cout << "Unable to open file" << strerror(errno) << std::endl;
  }

  return data;
}