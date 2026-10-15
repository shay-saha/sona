#include "analysis.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
std::vector<float> tone(double hz, int rate, double amplitude = 0.2) {
    std::vector<float> x(sona::frameSize);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = float(amplitude * std::sin(2 * sona::pi * hz * i / rate));
    return x;
}
void put16(std::vector<std::uint8_t> &b, std::uint16_t n) {
    b.push_back(n & 255);
    b.push_back(n >> 8);
}
void put32(std::vector<std::uint8_t> &b, std::uint32_t n) {
    for (int i = 0; i < 4; ++i)
        b.push_back((n >> (i * 8)) & 255);
}
void tag(std::vector<std::uint8_t> &b, const char *s) {
    b.insert(b.end(), s, s + 4);
}
std::vector<std::uint8_t> wave(int bits = 16, bool floating = false) {
    std::vector<std::uint8_t> b;
    const int channels = 2, width = bits / 8;
    tag(b, "RIFF");
    put32(b, 36 + std::uint32_t(sona::frameSize) * channels * width);
    tag(b, "WAVE");
    tag(b, "fmt ");
    put32(b, 16);
    put16(b, floating ? 3 : 1);
    put16(b, channels);
    put32(b, 24000);
    put32(b, 24000 * channels * width);
    put16(b, channels * width);
    put16(b, bits);
    tag(b, "data");
    put32(b, std::uint32_t(sona::frameSize) * channels * width);
    for (std::size_t i = 0; i < sona::frameSize; ++i)
        for (int c = 0; c < channels; ++c) {
            const float s = float(0.5 * std::sin(2 * sona::pi * 220 * i / 24000));
            if (floating) {
                std::uint32_t n;
                std::memcpy(&n, &s, 4);
                put32(b, n);
            } else if (bits == 16)
                put16(b, std::uint16_t(std::int16_t(s * 32767)));
            else if (bits == 24) {
                const auto n = std::uint32_t(std::int32_t(s * 8388607));
                for (int j = 0; j < 3; ++j)
                    b.push_back((n >> (8 * j)) & 255);
            } else
                put32(b, std::uint32_t(std::int32_t(double(s) * 2147483647)));
        }
    return b;
}
} // namespace

int main() {
    try {
        sona::Analyzer a;
        auto silent = a.analyze(std::vector<float>(sona::frameSize), 24000);
        check(!silent.active && silent.pitchHz == 0 && silent.rmsDb == -90, "Silence must have no pitch");
        check(!a.analyze(std::vector<float>(100), 24000).active, "Short frames must not be analyzed");
        check(!a.analyze(tone(220, 24000), 0).active, "Invalid sample rates must be rejected");
        for (int rate : {8000, 24000, 44100, 48000})
            for (double hz : {110.0, 220.0, 440.0}) {
                a.reset();
                auto f = a.analyze(tone(hz, rate), rate);
                check(f.active, "A tone must be active");
                check(std::abs(f.pitchHz - hz) < hz * 0.02, "Pitch must be within 2% of a known tone");
                check(std::abs(f.centroidHz - hz) < 20, "Centroid must follow a pure tone");
                check(f.periodicity > 0.9 && f.flatness < 0.01,
                      "A tone must be periodic and spectrally sparse");
                check(std::abs(f.rmsDb + 16.99) < 0.3, "RMS dBFS must follow digital amplitude");
            }
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> random(-0.5, 0.5);
        std::vector<float> noise(sona::frameSize);
        for (auto &v : noise)
            v = random(generator);
        a.reset();
        const auto noisy = a.analyze(noise, 24000);
        check(noisy.pitchHz == 0 && noisy.flatness > 0.3, "White noise must not produce stable pitch");
        auto dc = a.analyze(std::vector<float>(sona::frameSize, 0.8f), 24000);
        check(!dc.active, "DC offset must be removed");
        auto malformed = tone(220, 24000);
        malformed[3] = std::nanf("");
        malformed[4] = INFINITY;
        check(std::isfinite(a.analyze(malformed, 24000).rmsDb), "Non-finite input must be sanitized");
        a.reset();
        const auto warm = a.analyze(sona::synthesize(0, 0, sona::frameSize, 24000), 24000);
        a.reset();
        const auto bright = a.analyze(sona::synthesize(1, 0, sona::frameSize, 24000), 24000);
        check(bright.centroidHz > warm.centroidHz * 3 && bright.angularity > warm.angularity,
              "Synthetic contrasts must differ in brightness");
        for (int bits : {16, 24, 32}) {
            auto w = sona::decodeWave(wave(bits));
            check(bool(w) && w.sampleRate == 24000 && w.samples.size() == sona::frameSize,
                  "Integer WAV must decode and downmix");
            check(std::abs(a.analyze(w.samples, w.sampleRate).pitchHz - 220) < 2,
                  "Decoded WAV must preserve frequency");
        }
        check(bool(sona::decodeWave(wave(32, true))), "IEEE float32 WAV must decode");
        check(!sona::decodeWave({1, 2, 3}), "Non-WAV data must be rejected");
        auto truncated = wave();
        truncated.resize(60);
        check(!sona::decodeWave(truncated), "Truncated WAV must be rejected");
        auto corrupt = wave();
        corrupt[40] = 0xff;
        corrupt[41] = 0xff;
        corrupt[42] = 0xff;
        corrupt[43] = 0x7f;
        check(!sona::decodeWave(corrupt), "Oversized WAV chunks must be rejected");
        auto unsupported = wave();
        unsupported[20] = 6;
        check(!sona::decodeWave(unsupported), "Compressed WAV must be rejected");
        auto unaligned = wave();
        unaligned[32] = 1;
        check(!sona::decodeWave(unaligned), "Invalid alignment must be rejected");
        std::cout << "Signal analysis and WAV decoding checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
