#include "speaker_features.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
std::vector<float> tone(double hz, int rate, double seconds = 1) {
    std::vector<float> out(std::size_t(rate * seconds));
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = float(0.2 * std::sin(2 * sona::pi * hz * i / rate));
    return out;
}
} // namespace
int main() {
    try {
        const auto a = sona::extractSpeakerFeatures(tone(220, 16000), 16000);
        check(a.size() == sona::speakerFeatureNames().size() && a.size() == 46, "Descriptor schema mismatch");
        check(std::abs(a[4] - 220) < 2 && a[1] > 0.99 && a[2] > 0.99,
              "A periodic signal must retain its known pitch");
        check(a[8] > 25, "A clean tone must have high estimated harmonicity");
        const auto b = sona::extractSpeakerFeatures(tone(220, 44100), 44100);
        check(std::abs(a[4] - b[4]) < 1, "Resampling must preserve pitch");
        auto aliased = sona::resampleForDescriptor(tone(12000, 44100), 44100);
        double energy = 0;
        for (std::size_t i = 100; i + 100 < aliased.size(); ++i)
            energy += aliased[i] * aliased[i];
        check(std::sqrt(energy / (aliased.size() - 200)) < 0.005,
              "Downsampling must suppress out-of-band tones");
        auto silence = sona::extractSpeakerFeatures(std::vector<float>(16000), 16000);
        check(silence[1] == 0 && silence[2] == 0 && silence[4] == 0,
              "Silence must not create a speaker measurement");
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> random(-0.3, 0.3);
        std::vector<float> noise(16000);
        for (auto &v : noise)
            v = random(generator);
        auto noiseFeatures = sona::extractSpeakerFeatures(noise, 16000);
        for (auto &v : noise)
            v *= 0.25f;
        auto quieter = sona::extractSpeakerFeatures(noise, 16000);
        check(std::abs(noiseFeatures[17] - quieter[17] - 12.0412) < 0.05,
              "Digital level must track a known gain change");
        for (std::size_t i = 20; i < 46; ++i)
            check(std::abs(noiseFeatures[i] - quieter[i]) < 0.01,
                  "MFCC C1–C13 must reject a broadband uniform gain offset");
        for (double value : a)
            check(std::isfinite(value), "Descriptors must be finite");
        std::cout << "Descriptor schema, resampling, alias rejection, pitch, harmonicity, silence and gain "
                     "checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
