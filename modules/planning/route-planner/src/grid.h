#pragma once
#include <vector>

struct Point {
    int x, y;

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

struct PointHash {
    std::size_t operator()(const Point& point) const {
        return std::hash<int>()(point.x) ^ std::hash<int>()(point.y);
    }
};

class Grid {
public:
    Grid(int width, int height);
    std::vector<Point> getNeighbors(const Point& point) const;
    float getCost(const Point& a, const Point& b) const;
    bool isWalkable(const Point& point) const;

private:
    int width;
    int height;
    std::vector<std::vector<bool>> walkable;
};

