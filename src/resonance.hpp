#pragma once
#include <array>
#include <vector>

namespace sona {
struct Resonances {
    std::array<double, 4> frequencyHz{};
    std::array<double, 4> bandwidthHz{};
    bool valid = false;
};

// Fixed 16 kHz, 50 ms Burg-LPC analysis. These are acoustic estimates, not anatomy measurements.
Resonances estimateResonances(const std::vector<float> &samples);
} // namespace sona
