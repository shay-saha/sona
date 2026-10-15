#include "human_mesh.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace sona {
namespace {
double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
struct Part {
    Vec3 center, radii, end;
    bool capsule = false;

    double distance(Vec3 point) const {
        const auto local = point - center;
        if (capsule) {
            const auto axis = end - center;
            const double along = std::clamp(dot(local, axis) / dot(axis, axis), 0.0, 1.0);
            return (local - axis * along).length() - radii.x;
        }
        const Vec3 scaled{local.x / radii.x, local.y / radii.y, local.z / radii.z};
        const Vec3 gradient{scaled.x / radii.x, scaled.y / radii.y, scaled.z / radii.z};
        const double radius = scaled.length(), slope = gradient.length();
        return slope > 1e-12 ? radius * (radius - 1) / slope : -std::min({radii.x, radii.y, radii.z});
    }
};
class AnatomyField {
  public:
    AnatomyField() {
        ellipsoid({0, 0.94, 0}, {0.162, 0.13, 0.105});
        ellipsoid({0, 1.095, 0}, {0.126, 0.165, 0.087});
        ellipsoid({0, 1.265, -0.004}, {0.18, 0.154, 0.103});
        capsule({0, 1.365, 0}, {0, 1.49, 0}, 0.045);
        ellipsoid({0, 1.63, -0.006}, {0.084, 0.12, 0.084});
        ellipsoid({0, 1.55, 0.022}, {0.064, 0.057, 0.065});
        ellipsoid({0, 1.614, 0.079}, {0.016, 0.027, 0.026});
        for (double side : {-1.0, 1.0}) {
            ellipsoid({side * 0.084, 1.615, -0.007}, {0.012, 0.029, 0.019});
            capsule({side * 0.14, 1.335, 0}, {side * 0.218, 1.32, 0}, 0.057);
            capsule({side * 0.223, 1.305, 0}, {side * 0.274, 1.078, 0.006}, 0.043);
            ellipsoid({side * 0.245, 1.2, -0.002}, {0.047, 0.109, 0.046});
            capsule({side * 0.274, 1.078, 0.006}, {side * 0.318, 0.85, 0.025}, 0.032);
            ellipsoid({side * 0.291, 0.998, 0.012}, {0.038, 0.085, 0.035});
            ellipsoid({side * 0.323, 0.791, 0.032}, {0.033, 0.069, 0.023});
            capsule({side * 0.3, 0.822, 0.043}, {side * 0.285, 0.777, 0.056}, 0.012);
            capsule({side * 0.1, 0.92, 0}, {side * 0.112, 0.505, 0.009}, 0.065);
            ellipsoid({side * 0.11, 0.76, -0.006}, {0.078, 0.185, 0.077});
            ellipsoid({side * 0.112, 0.506, 0.012}, {0.051, 0.06, 0.051});
            capsule({side * 0.112, 0.505, 0.009}, {side * 0.115, 0.12, -0.005}, 0.037);
            ellipsoid({side * 0.115, 0.325, -0.016}, {0.054, 0.14, 0.054});
            ellipsoid({side * 0.115, 0.046, 0.054}, {0.058, 0.046, 0.125});
        }
    }

    double distance(Vec3 point) const {
        double result = 10;
        constexpr double blend = 0.022;
        for (const auto &part : parts_) {
            const double next = part.distance(point);
            const double overlap = std::max(blend - std::abs(result - next), 0.0) / blend;
            result = std::min(result, next) - overlap * overlap * blend * 0.25;
        }
        return result;
    }

    Vec3 normal(Vec3 point) const {
        constexpr double step = 0.0002;
        return Vec3{distance(point + Vec3{step, 0, 0}) - distance(point - Vec3{step, 0, 0}),
                    distance(point + Vec3{0, step, 0}) - distance(point - Vec3{0, step, 0}),
                    distance(point + Vec3{0, 0, step}) - distance(point - Vec3{0, 0, step})}
            .normalized();
    }

  private:
    void ellipsoid(Vec3 center, Vec3 radii) {
        parts_.push_back({center, radii, {}, false});
    }
    void capsule(Vec3 start, Vec3 end, double radius) {
        parts_.push_back({start, {radius, radius, radius}, end, true});
    }
    std::vector<Part> parts_;
};
} // namespace

HumanMesh makeHumanMesh(double cellSize) {
    if (!std::isfinite(cellSize) || cellSize < 0.012 || cellSize > 0.05)
        throw std::invalid_argument("Human mesh cell size must be 0.012–0.05 metres");
    const AnatomyField field;
    constexpr Vec3 minimum{-0.46, -0.055, -0.21}, maximum{0.46, 1.84, 0.24};
    const int nx = int(std::ceil((maximum.x - minimum.x) / cellSize)) + 1;
    const int ny = int(std::ceil((maximum.y - minimum.y) / cellSize)) + 1;
    const int nz = int(std::ceil((maximum.z - minimum.z) / cellSize)) + 1;
    auto index = [nx, nz](int x, int y, int z) { return std::uint32_t((y * nz + z) * nx + x); };
    auto point = [=](int x, int y, int z) {
        return minimum + Vec3{x * cellSize, y * cellSize, z * cellSize};
    };
    std::vector<double> distances(std::size_t(nx * ny * nz));
    for (int y = 0; y < ny; ++y)
        for (int z = 0; z < nz; ++z)
            for (int x = 0; x < nx; ++x) {
                double value = field.distance(point(x, y, z));
                distances[index(x, y, z)] = std::abs(value) < 1e-10 ? 1e-10 : value;
            }
    HumanMesh mesh;
    std::unordered_map<std::uint64_t, std::uint32_t> intersections;
    constexpr int corners[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                   {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    constexpr int tetrahedra[6][4] = {{0, 5, 1, 6}, {0, 1, 2, 6}, {0, 2, 3, 6},
                                      {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}};
    for (int y = 0; y < ny - 1; ++y)
        for (int z = 0; z < nz - 1; ++z)
            for (int x = 0; x < nx - 1; ++x) {
                std::array<std::uint32_t, 8> ids;
                std::array<Vec3, 8> positions;
                for (int corner = 0; corner < 8; ++corner) {
                    const auto *offset = corners[corner];
                    ids[corner] = index(x + offset[0], y + offset[1], z + offset[2]);
                    positions[corner] = point(x + offset[0], y + offset[1], z + offset[2]);
                }
                auto vertex = [&](int a, int b) {
                    const auto low = std::min(ids[a], ids[b]), high = std::max(ids[a], ids[b]);
                    const std::uint64_t key = (std::uint64_t(low) << 32) | high;
                    const auto existing = intersections.find(key);
                    if (existing != intersections.end())
                        return existing->second;
                    const double fraction = distances[ids[a]] / (distances[ids[a]] - distances[ids[b]]);
                    const auto position = positions[a] + (positions[b] - positions[a]) * fraction;
                    const auto result = std::uint32_t(mesh.positions.size());
                    mesh.positions.push_back(position);
                    mesh.normals.push_back(field.normal(position));
                    intersections.emplace(key, result);
                    return result;
                };
                auto triangle = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
                    const auto normal =
                        cross(mesh.positions[b] - mesh.positions[a], mesh.positions[c] - mesh.positions[a]);
                    if (dot(normal, mesh.normals[a] + mesh.normals[b] + mesh.normals[c]) < 0)
                        std::swap(b, c);
                    mesh.faces.push_back({a, b, c});
                };
                for (const auto &tetrahedron : tetrahedra) {
                    std::array<int, 4> inside{}, outside{};
                    int insideCount = 0, outsideCount = 0;
                    for (int corner : tetrahedron)
                        if (distances[ids[corner]] < 0)
                            inside[insideCount++] = corner;
                        else
                            outside[outsideCount++] = corner;
                    if (insideCount == 0 || insideCount == 4)
                        continue;
                    if (insideCount == 1)
                        triangle(vertex(inside[0], outside[0]), vertex(inside[0], outside[1]),
                                 vertex(inside[0], outside[2]));
                    else if (insideCount == 3)
                        triangle(vertex(outside[0], inside[0]), vertex(outside[0], inside[1]),
                                 vertex(outside[0], inside[2]));
                    else {
                        const auto a = vertex(inside[0], outside[0]), b = vertex(inside[0], outside[1]);
                        const auto c = vertex(inside[1], outside[1]), d = vertex(inside[1], outside[0]);
                        triangle(a, b, c);
                        triangle(a, c, d);
                    }
                }
            }
    std::set<std::array<std::uint32_t, 2>> edges;
    for (const auto &face : mesh.faces)
        for (int edge = 0; edge < 3; ++edge) {
            const auto a = face[edge], b = face[(edge + 1) % 3];
            edges.insert({std::min(a, b), std::max(a, b)});
        }
    mesh.edges.assign(edges.begin(), edges.end());
    return mesh;
}
} // namespace sona
