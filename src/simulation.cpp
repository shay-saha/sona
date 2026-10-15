#include "simulation.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace sona {
namespace {
double bounded(double v, double low, double high, double fallback) {
    return std::isfinite(v) ? std::clamp(v, low, high) : fallback;
}
} // namespace
double Vec3::length() const {
    return std::sqrt(x * x + y * y + z * z);
}
Vec3 Vec3::normalized() const {
    const double n = length();
    return n > 1e-12 ? *this * (1 / n) : Vec3{};
}

SurfaceMesh makeIcosphere(int subdivisions) {
    if (subdivisions < 0 || subdivisions > 6)
        throw std::invalid_argument("Icosphere subdivision must be 0–6");
    const double t = (1 + std::sqrt(5.0)) / 2;
    SurfaceMesh m;
    m.directions = {{-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
                    {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1}};
    for (auto &v : m.directions)
        v = v.normalized();
    m.faces = {{0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
               {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
               {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};
    for (int level = 0; level < subdivisions; ++level) {
        std::map<std::array<std::uint32_t, 2>, std::uint32_t> cache;
        auto midpoint = [&](std::uint32_t a, std::uint32_t b) {
            const std::array<std::uint32_t, 2> key{std::min(a, b), std::max(a, b)};
            auto found = cache.find(key);
            if (found != cache.end())
                return found->second;
            const auto index = std::uint32_t(m.directions.size());
            m.directions.push_back((m.directions[a] + m.directions[b]).normalized());
            cache.emplace(key, index);
            return index;
        };
        std::vector<std::array<std::uint32_t, 3>> next;
        next.reserve(m.faces.size() * 4);
        for (auto f : m.faces) {
            const auto a = midpoint(f[0], f[1]), b = midpoint(f[1], f[2]), c = midpoint(f[2], f[0]);
            next.insert(next.end(), {{f[0], a, c}, {f[1], b, a}, {f[2], c, b}, {a, b, c}});
        }
        m.faces = std::move(next);
    }
    std::map<std::array<std::uint32_t, 2>, bool> edges;
    for (const auto &f : m.faces)
        for (int i = 0; i < 3; ++i) {
            const auto a = f[std::size_t(i)], b = f[std::size_t((i + 1) % 3)];
            edges.emplace(std::array<std::uint32_t, 2>{std::min(a, b), std::max(a, b)}, true);
        }
    m.neighbors.resize(m.directions.size());
    for (auto e : edges) {
        m.edges.push_back(e.first);
        m.neighbors[e.first[0]].push_back(e.first[1]);
        m.neighbors[e.first[1]].push_back(e.first[0]);
    }
    return m;
}

SurfaceSimulation::SurfaceSimulation(int subdivisions)
    : mesh_(makeIcosphere(subdivisions)), displacement_(mesh_.directions.size()),
      velocity_(displacement_.size()), acceleration_(displacement_.size()), forcing_(displacement_.size()) {}

void SurfaceSimulation::setParameters(SimulationParameters p) {
    parameters_ = {bounded(p.waveSpeed, 1, 60, 18), bounded(p.damping, 0, 20, 2.4),
                   bounded(p.restoring, 1, 100, 14), bounded(p.gain, 0, 3, 1)};
}
void SurfaceSimulation::setDrive(SurfaceDrive d) {
    drive_ = {bounded(d.amplitude, 0, 1, 0), bounded(d.angularity, 0, 1, 0),
              bounded(d.frequency, 0.2, 3, 0.8)};
    const double lobes = 3 + std::round(drive_.angularity * 6);
    double mean = 0;
    for (std::size_t i = 0; i < forcing_.size(); ++i) {
        const auto v = mesh_.directions[i];
        const double longitude = std::atan2(v.z, v.x), latitude = std::acos(std::clamp(v.y, -1.0, 1.0));
        const double a = std::cos(lobes * longitude) * std::pow(std::max(0.0, 1 - v.y * v.y), 1.2);
        forcing_[i] = std::copysign(std::pow(std::abs(a), 1 + drive_.angularity * 3), a) +
                      0.38 * std::cos((lobes - 1) * latitude);
        mean += forcing_[i];
    }
    mean /= double(forcing_.size());
    for (auto &v : forcing_)
        v -= mean;
}
void SurfaceSimulation::advance(double elapsed) {
    if (!std::isfinite(elapsed) || elapsed <= 0)
        return;
    // Discard excess wall time after stalls; the solver always uses its fixed step.
    accumulator_ += std::min(elapsed, 0.1);
    while (accumulator_ + 1e-12 >= timeStep) {
        step();
        accumulator_ -= timeStep;
    }
}
void SurfaceSimulation::singleStep() {
    step();
}
void SurfaceSimulation::reset() {
    std::fill(displacement_.begin(), displacement_.end(), 0);
    std::fill(velocity_.begin(), velocity_.end(), 0);
    std::fill(acceleration_.begin(), acceleration_.end(), 0);
    time_ = 0;
    accumulator_ = 0;
}
void SurfaceSimulation::impulse(std::size_t i, double value) {
    if (i < velocity_.size() && std::isfinite(value))
        velocity_[i] += std::clamp(value, -5.0, 5.0);
}
void SurfaceSimulation::step() {
    const double wave = parameters_.waveSpeed * parameters_.waveSpeed;
    const double excitation = 0.65 + 0.35 * std::sin(6.283185307179586 * drive_.frequency * time_);
    for (std::size_t i = 0; i < displacement_.size(); ++i) {
        double laplacian = 0;
        for (auto j : mesh_.neighbors[i])
            laplacian += displacement_[j] - displacement_[i];
        acceleration_[i] = wave * laplacian - parameters_.restoring * displacement_[i] -
                           parameters_.damping * velocity_[i] +
                           14 * parameters_.gain * drive_.amplitude * forcing_[i] * excitation;
    }
    for (std::size_t i = 0; i < displacement_.size(); ++i) {
        velocity_[i] += acceleration_[i] * timeStep;
        displacement_[i] += velocity_[i] * timeStep;
        // Hard travel stops keep interactive gain changes from folding the radial surface.
        if (std::abs(displacement_[i]) > 0.72) {
            displacement_[i] = std::clamp(displacement_[i], -0.72, 0.72);
            velocity_[i] *= -0.15;
        }
    }
    time_ += timeStep;
}
double SurfaceSimulation::energy() const {
    double e = 0;
    for (std::size_t i = 0; i < velocity_.size(); ++i)
        e += 0.5 * velocity_[i] * velocity_[i] +
             0.5 * parameters_.restoring * displacement_[i] * displacement_[i];
    for (auto edge : mesh_.edges) {
        const double d = displacement_[edge[0]] - displacement_[edge[1]];
        e += 0.5 * parameters_.waveSpeed * parameters_.waveSpeed * d * d;
    }
    return e;
}
bool SurfaceSimulation::finite() const {
    for (std::size_t i = 0; i < velocity_.size(); ++i)
        if (!std::isfinite(displacement_[i]) || !std::isfinite(velocity_[i]) ||
            std::abs(displacement_[i]) > 0.721)
            return false;
    return true;
}
} // namespace sona
