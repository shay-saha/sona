#include "resonance.hpp"
#include "speaker_features.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
std::vector<float> vowel(int pitch) {
    constexpr int rate = 16000;
    std::vector<double> signal(rate * 2);
    for (std::size_t i = 0; i < signal.size(); ++i)
        signal[i] = i % (rate / pitch) == 0 ? 1 : 0;
    const std::array<int, 5> frequencies{500, 1500, 2500, 3500, 4500};
    const std::array<int, 5> bandwidths{100, 150, 180, 200, 220};
    for (std::size_t pole = 0; pole < frequencies.size(); ++pole) {
        const double radius = std::exp(-sona::pi * bandwidths[pole] / rate);
        const double feedback = 2 * radius * std::cos(2 * sona::pi * frequencies[pole] / rate);
        double previous = 0, older = 0;
        for (auto &sample : signal) {
            const double output = sample + feedback * previous - radius * radius * older;
            older = previous;
            previous = output;
            sample = output;
        }
    }
    double peak = 0;
    for (double sample : signal)
        peak = std::max(peak, std::abs(sample));
    std::vector<float> result;
    for (double sample : signal)
        result.push_back(float(0.5 * sample / peak));
    return result;
}
} // namespace
int main() {
    try {
        for (int pitch : {100, 180}) {
            const auto audio = vowel(pitch);
            const auto resonance = sona::estimateResonances(audio);
            check(resonance.valid, "Known vowel resonances must be recoverable");
            for (std::size_t i = 0; i < 4; ++i)
                check(std::abs(resonance.frequencyHz[i] - (500 + 1000 * i)) < 100,
                      "Estimated formant must be close to the known filter resonance");
            const auto legacy = sona::extractSpeakerFeatures(audio, 16000);
            const auto extended = sona::extractExtendedSpeakerFeatures(audio, 16000);
            check(extended.size() == sona::extendedSpeakerFeatureNames().size(), "Extended schema mismatch");
            check(std::equal(legacy.begin(), legacy.end(), extended.begin()),
                  "Legacy descriptor values changed");
            auto quiet = audio;
            for (auto &sample : quiet)
                sample *= 0.1f;
            const auto reduced = sona::estimateResonances(quiet);
            check(reduced.valid, "Gain changes must not invalidate a clean vowel");
            check(std::abs(reduced.frequencyHz[0] - resonance.frequencyHz[0]) < 1,
                  "Gain changes must not shift the estimated resonance");
        }
        check(!sona::estimateResonances(std::vector<float>(2048)).valid, "Silence has no valid resonances");
        check(!sona::estimateResonances(std::vector<float>(50)).valid, "Short input must be rejected");
        std::cout << "Synthetic vowel resonances, gain stability, and legacy feature parity passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
