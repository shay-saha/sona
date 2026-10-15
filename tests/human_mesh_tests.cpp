#include "human_mesh.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace
int main() {
    try {
        const auto mesh = sona::makeHumanMesh();
        check(mesh.positions.size() > 5000 && mesh.faces.size() > 10000, "Human surface is undersampled");
        check(mesh.normals.size() == mesh.positions.size(), "Every vertex needs a surface normal");
        double minimumY = 10, maximumY = -10;
        for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
            const auto p = mesh.positions[i], n = mesh.normals[i];
            check(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z), "Non-finite human vertex");
            check(std::abs(n.length() - 1) < 1e-8, "Human normals must be unit length");
            minimumY = std::min(minimumY, p.y);
            maximumY = std::max(maximumY, p.y);
        }
        check(std::abs(minimumY) < 0.01 && std::abs(maximumY - sona::referenceHumanHeight) < 0.01,
              "Generic figure must stand on the floor at its reference height");
        std::map<std::array<std::uint32_t, 2>, int> uses;
        for (const auto &face : mesh.faces) {
            for (auto index : face)
                check(index < mesh.positions.size(), "Human triangle has an invalid vertex");
            check(face[0] != face[1] && face[1] != face[2] && face[0] != face[2], "Collapsed triangle");
            for (int edge = 0; edge < 3; ++edge) {
                const auto a = face[edge], b = face[(edge + 1) % 3];
                ++uses[{std::min(a, b), std::max(a, b)}];
            }
        }
        for (const auto &[edge, count] : uses) {
            (void)edge;
            check(count == 2, "Human surface must be closed without boundary or nonmanifold edges");
        }
        check(uses.size() == mesh.edges.size(), "Wire edges must match the triangle surface");
        bool rejected = false;
        try {
            sona::makeHumanMesh(0);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "Invalid voxel resolution must be rejected");
        std::cout << mesh.positions.size() << " human vertices, " << mesh.faces.size()
                  << " closed triangles\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
