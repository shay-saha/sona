#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace sona {
struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3 operator+(Vec3 b) const {
        return {x + b.x, y + b.y, z + b.z};
    }
    Vec3 operator-(Vec3 b) const {
        return {x - b.x, y - b.y, z - b.z};
    }
    Vec3 operator*(double s) const {
        return {x * s, y * s, z * s};
    }
    double length() const;
    Vec3 normalized() const;
};
struct SurfaceMesh {
    std::vector<Vec3> directions;
    std::vector<std::array<std::uint32_t, 3>> faces;
    std::vector<std::array<std::uint32_t, 2>> edges;
    std::vector<std::vector<std::uint32_t>> neighbors;
};
SurfaceMesh makeIcosphere(int subdivisions);

struct SimulationParameters {
    double waveSpeed = 18;
    double damping = 2.4;
    double restoring = 14;
    double gain = 1;
};
struct SurfaceDrive {
    double amplitude = 0;
    double angularity = 0;
    double frequency = 0.8; // Slow excitation rate, not the audio carrier frequency.
};
class SurfaceSimulation {
  public:
    static constexpr double timeStep = 1.0 / 240.0;
    explicit SurfaceSimulation(int subdivisions = 5);
    void setParameters(SimulationParameters parameters);
    void setDrive(SurfaceDrive drive);
    void advance(double elapsedSeconds);
    void singleStep();
    void reset();
    void impulse(std::size_t vertex, double velocity);
    double energy() const;
    bool finite() const;
    double time() const {
        return time_;
    }
    const SurfaceMesh &mesh() const {
        return mesh_;
    }
    const std::vector<double> &displacement() const {
        return displacement_;
    }
    const std::vector<double> &velocity() const {
        return velocity_;
    }
    SimulationParameters parameters() const {
        return parameters_;
    }

  private:
    void step();
    SurfaceMesh mesh_;
    SimulationParameters parameters_;
    SurfaceDrive drive_;
    std::vector<double> displacement_, velocity_, acceleration_, forcing_;
    double time_ = 0, accumulator_ = 0;
};
} // namespace sona
