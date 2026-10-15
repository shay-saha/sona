#include "analysis.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>

namespace sona {
namespace {
using Complex = std::complex<double>;
void fft(std::array<Complex, frameSize> &a) {
    for (std::size_t i = 1, j = 0; i < frameSize; ++i) {
        std::size_t bit = frameSize >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
    for (std::size_t length = 2; length <= frameSize; length <<= 1) {
        const Complex root = std::polar(1.0, -2.0 * pi / double(length));
        for (std::size_t i = 0; i < frameSize; i += length) {
            Complex w(1, 0);
            for (std::size_t j = 0; j < length / 2; ++j) {
                auto u = a[i + j], v = a[i + j + length / 2] * w;
                a[i + j] = u + v;
                a[i + j + length / 2] = u - v;
                w *= root;
            }
        }
    }
}
std::uint16_t u16(const std::uint8_t *p) {
    return std::uint16_t(p[0]) | std::uint16_t(p[1]) << 8;
}
std::uint32_t u32(const std::uint8_t *p) {
    return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16 |
           std::uint32_t(p[3]) << 24;
}
} // namespace

void Analyzer::reset() {
    previous_.fill(0);
    hasPrevious_ = false;
}

Features Analyzer::analyze(const std::vector<float> &samples, int sampleRate) {
    Features f;
    if (samples.size() < frameSize || sampleRate < 8000 || sampleRate > 192000)
        return f;
    std::array<double, frameSize> x{};
    const auto start = samples.size() - frameSize;
    double mean = 0;
    for (std::size_t i = 0; i < frameSize; ++i) {
        const double value = samples[start + i];
        x[i] = std::isfinite(value) ? std::clamp(value, -1.0, 1.0) : 0;
        mean += x[i];
    }
    mean /= frameSize;
    double energy = 0;
    int crossings = 0;
    for (std::size_t i = 0; i < frameSize; ++i) {
        x[i] -= mean;
        energy += x[i] * x[i];
        if (i > 0 && ((x[i] >= 0) != (x[i - 1] >= 0)))
            ++crossings;
    }
    f.rmsDb = std::max(-90.0, 10 * std::log10(energy / frameSize + 1e-12));
    f.active = f.rmsDb > -60;
    f.crossingRate = double(crossings) / (frameSize - 1);
    for (std::size_t i = 0; i < f.waveform.size(); ++i)
        f.waveform[i] = float(x[i * 8]);
    if (!f.active) {
        reset();
        return f;
    }

    std::array<Complex, frameSize> bins{};
    for (std::size_t i = 0; i < frameSize; ++i)
        bins[i] = x[i] * (0.5 - 0.5 * std::cos(2 * pi * i / (frameSize - 1)));
    fft(bins);
    std::array<double, frameSize / 2 + 1> magnitude{};
    double total = 0, weighted = 0, logPower = 0, power = 0, positiveChange = 0;
    for (std::size_t i = 1; i < magnitude.size(); ++i) {
        magnitude[i] = std::abs(bins[i]) / (frameSize / 4.0);
        total += magnitude[i];
        weighted += magnitude[i] * (double(i) * sampleRate / frameSize);
        const double p = magnitude[i] * magnitude[i] + 1e-16;
        logPower += std::log(p);
        power += p;
    }
    f.centroidHz = weighted / std::max(total, 1e-12);
    f.flatness = std::clamp(std::exp(logPower / (frameSize / 2)) / (power / (frameSize / 2)), 0.0, 1.0);
    // A fixed acoustic representation shared by offline research and native inference.
    constexpr int melBands = 26;
    std::array<double, melBands + 2> melEdges{};
    std::array<double, melBands> logMel{};
    const double melLow = 2595 * std::log10(1 + 40.0 / 700);
    const double melHigh = 2595 * std::log10(1 + std::min(7600.0, sampleRate * 0.5) / 700);
    for (int i = 0; i < melBands + 2; ++i)
        melEdges[std::size_t(i)] =
            700 * (std::pow(10.0, (melLow + (melHigh - melLow) * i / (melBands + 1)) / 2595) - 1);
    for (int b = 0; b < melBands; ++b) {
        double sum = 0;
        for (std::size_t k = 1; k < magnitude.size(); ++k) {
            const double hz = double(k) * sampleRate / frameSize;
            const double weight = hz < melEdges[std::size_t(b + 1)]
                                      ? (hz - melEdges[std::size_t(b)]) /
                                            (melEdges[std::size_t(b + 1)] - melEdges[std::size_t(b)])
                                      : (melEdges[std::size_t(b + 2)] - hz) /
                                            (melEdges[std::size_t(b + 2)] - melEdges[std::size_t(b + 1)]);
            sum += magnitude[k] * magnitude[k] * std::max(0.0, weight);
        }
        logMel[std::size_t(b)] = std::log(std::max(sum, 1e-12));
    }
    for (std::size_t c = 0; c < f.mfcc.size(); ++c)
        for (int b = 0; b < melBands; ++b)
            f.mfcc[c] += std::sqrt(2.0 / melBands) * logMel[std::size_t(b)] *
                         std::cos(pi * (c + 1) * (b + 0.5) / melBands);
    for (std::size_t i = 1; i < magnitude.size(); ++i) {
        const double normalized = magnitude[i] / std::max(total, 1e-12);
        if (hasPrevious_)
            positiveChange += std::max(0.0, normalized - previous_[i]);
        previous_[i] = normalized;
    }
    f.flux = positiveChange;
    hasPrevious_ = true;
    const double high = std::min(12000.0, sampleRate * 0.5);
    for (std::size_t b = 0; b < spectrumBands; ++b) {
        const double loHz = 60 * std::pow(high / 60, double(b) / spectrumBands);
        const double hiHz = 60 * std::pow(high / 60, double(b + 1) / spectrumBands);
        const auto lo =
            std::clamp(std::size_t(loHz * frameSize / sampleRate), std::size_t(1), magnitude.size() - 1);
        const auto hi =
            std::clamp(std::size_t(std::ceil(hiHz * frameSize / sampleRate)), lo + 1, magnitude.size());
        double peak = 0;
        for (auto i = lo; i < hi; ++i)
            peak = std::max(peak, magnitude[i]);
        f.spectrum[b] = float(std::clamp((20 * std::log10(peak + 1e-8) + 80) / 80, 0.0, 1.0));
    }

    // YIN difference function and cumulative mean normalization.
    constexpr std::size_t window = frameSize / 2;
    const int minTau = std::max(2, sampleRate / 500);
    const int maxTau = std::min(int(window) - 2, sampleRate / 65);
    std::vector<double> cmnd(std::size_t(maxTau + 2), 1);
    double running = 0;
    for (int tau = 1; tau <= maxTau + 1; ++tau) {
        double diff = 0;
        for (std::size_t j = 0; j < window; ++j) {
            const double d = x[j] - x[j + std::size_t(tau)];
            diff += d * d;
        }
        running += diff;
        cmnd[std::size_t(tau)] = running > 1e-15 ? diff * tau / running : 1;
    }
    int best = 0;
    for (int tau = minTau; tau <= maxTau; ++tau) {
        if (cmnd[std::size_t(tau)] < 0.15) {
            while (tau < maxTau && cmnd[std::size_t(tau + 1)] < cmnd[std::size_t(tau)])
                ++tau;
            best = tau;
            break;
        }
    }
    if (best != 0) {
        const double a = cmnd[std::size_t(best - 1)], b = cmnd[std::size_t(best)],
                     c = cmnd[std::size_t(best + 1)];
        const double denominator = a - 2 * b + c;
        const double adjustment =
            std::abs(denominator) > 1e-12 ? std::clamp(0.5 * (a - c) / denominator, -1.0, 1.0) : 0;
        f.pitchHz = sampleRate / (best + adjustment);
        f.periodicity = std::clamp(1 - b, 0.0, 1.0);
        double dot = 0, aEnergy = 0, bEnergy = 0;
        for (std::size_t j = 0; j + std::size_t(best) < frameSize; ++j) {
            const double aValue = x[j], bValue = x[j + std::size_t(best)];
            dot += aValue * bValue;
            aEnergy += aValue * aValue;
            bEnergy += bValue * bValue;
        }
        const double correlation = dot / std::sqrt(std::max(1e-24, aEnergy * bEnergy));
        if (correlation > 0 && f.periodicity >= 0.85) {
            const double r = std::clamp(correlation, 1e-6, 1 - 1e-6);
            f.harmonicityDb = std::clamp(10 * std::log10(r / (1 - r)), -60.0, 60.0);
            f.harmonicityValid = true;
        }
    }
    const double brightness = std::clamp((f.centroidHz - 350) / 3500, 0.0, 1.0);
    f.angularity = std::clamp(0.75 * brightness + 0.15 * std::sqrt(f.flatness) + 0.1 * f.flux, 0.0, 1.0);
    return f;
}

std::vector<float> synthesize(int profile, std::size_t offset, std::size_t count, int sampleRate) {
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double t = double(offset + i) / sampleRate;
        double s = 0;
        if (profile == 0) {
            const double phase = 2 * pi * 164 * t + 0.55 * std::sin(2 * pi * 0.6 * t);
            for (int h = 1; h <= 12; ++h)
                s += std::sin(h * phase) * std::exp(-0.46 * (h - 1));
            s *= (0.26 + 0.035 * std::sin(2 * pi * 0.3 * t)) / 1.8;
        } else if (profile == 1) {
            const double phase = 2 * pi * 294 * t + 0.2 * std::sin(2 * pi * 0.8 * t);
            for (int h = 1; h <= 30; ++h)
                s += std::sin(h * phase) * (0.1 + 0.85 * std::exp(-std::pow((h - 14.0) / 8.0, 2)));
            s *= 0.075 * (0.7 + 0.3 * std::pow(std::sin(2 * pi * 1.4 * t), 2));
        } else {
            const double envelope = std::pow(std::max(0.0, std::sin(2 * pi * 4.2 * t)), 2);
            const double phase = 2 * pi * 205 * t + 4 * std::sin(2 * pi * 1.1 * t);
            for (int h = 1; h <= 16; ++h)
                s += std::sin(h * phase) * std::exp(-0.28 * (h - 1));
            s *= 0.19 * envelope;
        }
        out[i] = float(std::clamp(s, -0.95, 0.95));
    }
    return out;
}

WaveFile decodeWave(const std::vector<std::uint8_t> &bytes) {
    auto fail = [](const char *message) {
        WaveFile w;
        w.error = message;
        return w;
    };
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) || std::memcmp(bytes.data() + 8, "WAVE", 4))
        return fail("Choose a little-endian RIFF/WAVE file.");
    const std::size_t riffEnd = std::size_t(u32(bytes.data() + 4)) + 8;
    if (riffEnd > bytes.size() || riffEnd < 12)
        return fail("The WAV file is truncated.");
    int format = 0, channels = 0, rate = 0, bits = 0, alignment = 0;
    std::size_t dataOffset = 0, dataSize = 0;
    for (std::size_t pos = 12; pos + 8 <= riffEnd;) {
        const auto *p = bytes.data() + pos;
        const std::size_t size = u32(p + 4);
        if (size > riffEnd - pos - 8)
            return fail("A WAV chunk extends past the end of the file.");
        if (!std::memcmp(p, "fmt ", 4) && size >= 16) {
            format = u16(p + 8);
            channels = u16(p + 10);
            rate = int(u32(p + 12));
            alignment = u16(p + 20);
            bits = u16(p + 22);
        } else if (!std::memcmp(p, "data", 4) && dataOffset == 0) {
            dataOffset = pos + 8;
            dataSize = size;
        }
        pos += 8 + size + (size & 1);
    }
    if (channels < 1 || channels > 8 || rate < 8000 || rate > 192000 || !dataOffset || !dataSize)
        return fail("The WAV file needs valid audio data, 1–8 channels, and an 8–192 kHz sample rate.");
    if (!((format == 1 && (bits == 16 || bits == 24 || bits == 32)) || (format == 3 && bits == 32)))
        return fail("Supported WAV formats: PCM 16/24/32-bit or IEEE float32. Convert compressed or "
                    "extensible WAV first.");
    const int width = bits / 8;
    if (alignment != channels * width || dataSize % std::size_t(alignment) != 0)
        return fail("The WAV sample alignment is invalid.");
    WaveFile result;
    result.sampleRate = rate;
    const auto frames = dataSize / std::size_t(alignment);
    if (frames < frameSize)
        return fail("Choose a WAV file with at least 2048 audio frames.");
    result.samples.reserve(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        double sum = 0;
        for (int c = 0; c < channels; ++c) {
            const auto *p = bytes.data() + dataOffset + i * std::size_t(alignment) + std::size_t(c * width);
            double value;
            if (format == 3) {
                const auto raw = u32(p);
                float f;
                std::memcpy(&f, &raw, 4);
                value = std::isfinite(f) ? f : 0;
            } else if (bits == 16) {
                const std::uint32_t raw = u16(p);
                value = (raw >= 0x8000 ? double(raw) - 65536 : double(raw)) / 32768;
            } else if (bits == 24) {
                const auto raw = std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16;
                value = (raw >= 0x800000 ? double(raw) - 16777216 : double(raw)) / 8388608;
            } else {
                const auto raw = u32(p);
                value = (raw >= 0x80000000U ? double(raw) - 4294967296.0 : double(raw)) / 2147483648.0;
            }
            sum += std::clamp(value, -1.0, 1.0);
        }
        result.samples.push_back(float(sum / channels));
    }
    return result;
}
} // namespace sona
