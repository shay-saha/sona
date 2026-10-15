#pragma once
#include "simulation.hpp"

namespace sona {
constexpr double referenceHumanHeight = 1.75;

struct HumanMesh {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<std::array<std::uint32_t, 3>> faces;
    std::vector<std::array<std::uint32_t, 2>> edges;
};

// Generic anatomy in metres. Audio-derived stature is applied by the renderer.
HumanMesh makeHumanMesh(double cellSize = 0.018);
} // namespace sona
