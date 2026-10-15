#pragma once
#include "simulation.hpp"

namespace sona {
constexpr double referenceHumanHeight = 1.75;
constexpr double referenceHumanShoulders = 0.436;

struct HumanMesh {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<std::array<std::uint32_t, 3>> faces;
    std::vector<std::array<std::uint32_t, 2>> edges;
};

// Generic anatomy in metres. Audio-derived stature is applied by the renderer.
HumanMesh makeHumanMesh(double cellSize = 0.018);
HumanMesh withShoulderScale(const HumanMesh &reference, double factor);
} // namespace sona
