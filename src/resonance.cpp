#include "resonance.hpp"
#include "analysis.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

namespace sona {
Resonances estimateResonances(const std::vector<float> &samples) {
    constexpr int length = 800, order = 18, rate = 16000;
    Resonances result;
    if (samples.size() < length)
        return result;
    const auto offset = (samples.size() - length) / 2;
    double mean = 0;
    for (int i = 0; i < length; ++i) {
        if (!std::isfinite(samples[offset + i]))
            return result;
        mean += samples[offset + i];
    }
    mean /= length;
    const double preemphasis = std::exp(-2 * pi * 50 / rate);
    std::array<double, length> forward{}, backward{};
    double energy = 0;
    for (int i = 0; i < length; ++i) {
        const double current = samples[offset + i] - mean;
        const double previous = i > 0 ? samples[offset + i - 1] - mean : current;
        const double position = (i - (length - 1) * 0.5) / (length * 0.18);
        forward[i] = (current - preemphasis * previous) * std::exp(-0.5 * position * position);
        backward[i] = forward[i];
        energy += forward[i] * forward[i];
    }
    if (energy < 1e-12)
        return result;
    std::array<double, order + 1> coefficients{};
    coefficients[0] = 1;
    for (int degree = 1; degree <= order; ++degree) {
        double numerator = 0, denominator = 0;
        for (int i = degree; i < length; ++i) {
            numerator += forward[i] * backward[i - 1];
            denominator += forward[i] * forward[i] + backward[i - 1] * backward[i - 1];
        }
        if (denominator < 1e-24)
            return result;
        const double reflection = std::clamp(-2 * numerator / denominator, -0.9999, 0.9999);
        const auto previous = coefficients;
        coefficients[degree] = reflection;
        for (int i = 1; i < degree; ++i)
            coefficients[i] = previous[i] + reflection * previous[degree - i];
        for (int i = length - 1; i >= degree; --i) {
            const double oldForward = forward[i], oldBackward = backward[i - 1];
            forward[i] = oldForward + reflection * oldBackward;
            backward[i] = oldBackward + reflection * oldForward;
        }
    }
    using Complex = std::complex<double>;
    std::array<Complex, order> roots;
    for (int i = 0; i < order; ++i)
        roots[i] = std::polar(0.95, 2 * pi * (i + 0.25) / order);
    bool converged = false;
    for (int iteration = 0; iteration < 120; ++iteration) {
        const auto previous = roots;
        double maximumChange = 0;
        for (int i = 0; i < order; ++i) {
            Complex value = coefficients[0], denominator = 1;
            for (int degree = 1; degree <= order; ++degree)
                value = value * previous[i] + coefficients[degree];
            for (int j = 0; j < order; ++j)
                if (i != j)
                    denominator *= previous[i] - previous[j];
            if (std::abs(denominator) < 1e-30)
                return result;
            const Complex change = value / denominator;
            roots[i] = previous[i] - change;
            if (!std::isfinite(roots[i].real()) || !std::isfinite(roots[i].imag()))
                return result;
            maximumChange = std::max(maximumChange, std::abs(change));
        }
        if (maximumChange < 1e-7) {
            converged = true;
            break;
        }
    }
    if (!converged)
        return result;
    std::vector<std::pair<double, double>> candidates;
    for (const auto &root : roots) {
        if (root.imag() <= 0 || std::abs(root) <= 0 || std::abs(root) >= 1)
            continue;
        const double frequency = std::arg(root) * rate / (2 * pi);
        const double bandwidth = -std::log(std::abs(root)) * rate / pi;
        if (frequency >= 150 && frequency <= 5500 && bandwidth >= 30 && bandwidth <= 700)
            candidates.emplace_back(frequency, bandwidth);
    }
    std::sort(candidates.begin(), candidates.end());
    if (candidates.size() < 4)
        return result;
    for (int i = 0; i < 4; ++i) {
        result.frequencyHz[i] = candidates[i].first;
        result.bandwidthHz[i] = candidates[i].second;
    }
    const auto &f = result.frequencyHz;
    result.valid = f[0] <= 1300 && f[1] >= 600 && f[1] <= 3300 && f[2] >= 1400 && f[3] >= 2300;
    return result;
}
} // namespace sona
