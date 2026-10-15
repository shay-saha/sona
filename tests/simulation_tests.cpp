#include "simulation.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
} // namespace
int main() {
    try {
        auto mesh = sona::makeIcosphere(3);
        check(mesh.directions.size() == 642 && mesh.faces.size() == 1280 && mesh.edges.size() == 1920,
              "Icosphere topology is incorrect");
        check(mesh.directions.size() + mesh.faces.size() == mesh.edges.size() + 2,
              "Mesh must be a closed spherical manifold");
        for (std::size_t i = 0; i < mesh.directions.size(); ++i) {
            check(std::abs(mesh.directions[i].length() - 1) < 1e-12, "Rest directions must be unit length");
            check(mesh.neighbors[i].size() == 5 || mesh.neighbors[i].size() == 6, "Unexpected vertex degree");
            for (auto j : mesh.neighbors[i])
                check(std::find(mesh.neighbors[j].begin(), mesh.neighbors[j].end(), i) !=
                          mesh.neighbors[j].end(),
                      "Spring connectivity must be symmetric");
        }
        sona::SurfaceSimulation rest(3);
        rest.advance(0.1);
        check(rest.energy() == 0 && rest.finite(), "An unforced surface must remain at equilibrium");
        rest.impulse(0, 1);
        const double initial = rest.energy();
        rest.singleStep();
        check(rest.displacement()[0] > 0, "An impulse must move its vertex");
        rest.singleStep();
        check(std::abs(rest.displacement()[mesh.neighbors[0][0]]) > 0,
              "Motion must propagate through the springs");
        for (int i = 0; i < 2400; ++i)
            rest.singleStep();
        check(rest.energy() < initial * 1e-5 && rest.finite(), "Damping must dissipate unforced motion");
        sona::SurfaceSimulation a(2), b(2);
        a.impulse(0, 1);
        b.impulse(0, 1);
        for (int i = 0; i < 60; ++i)
            a.advance(1.0 / 60);
        for (int i = 0; i < 240; ++i)
            b.advance(1.0 / 240);
        check(std::abs(a.time() - b.time()) < 1e-12, "Fixed time must be independent of render cadence");
        for (std::size_t i = 0; i < a.displacement().size(); ++i)
            check(std::abs(a.displacement()[i] - b.displacement()[i]) < 1e-12,
                  "Solver states must agree across render rates");
        a.advance(std::numeric_limits<double>::quiet_NaN());
        a.advance(-2);
        check(a.finite(), "Invalid elapsed times must be ignored");
        const auto before = a.time();
        a.advance(1000);
        check(a.time() - before < 0.101, "A stall must not trigger unbounded catch-up");
        a.setParameters({60, 0, 100, 3});
        a.setDrive({1, 1, 3});
        for (int i = 0; i < 4800; ++i)
            a.singleStep();
        check(a.finite(), "Extreme interactive settings must remain bounded");
        a.setParameters({NAN, NAN, NAN, NAN});
        a.setDrive({NAN, NAN, NAN});
        for (int i = 0; i < 240; ++i)
            a.singleStep();
        check(a.finite(), "Non-finite controls must be sanitized");
        a.reset();
        check(a.energy() == 0 && a.time() == 0, "Reset must clear velocity, displacement, and time");
        bool rejected = false;
        try {
            sona::makeIcosphere(9);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "Unbounded mesh allocations must be rejected");
        std::cout
            << "Surface topology, propagation, damping, fixed-step determinism, bounds, and reset passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
