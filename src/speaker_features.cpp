#include "speaker_features.hpp"
#include "resonance.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sona {
namespace {
double mean(const std::vector<double> &v) {
    return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
double deviation(const std::vector<double> &v) {
    if (v.empty())
        return 0;
    const double m = mean(v);
    double sum = 0;
    for (double x : v)
        sum += (x - m) * (x - m);
    return std::sqrt(sum / v.size());
}
double quantile(std::vector<double> v, double q) {
    if (v.empty())
        return 0;
    std::sort(v.begin(), v.end());
    const double index = q * (v.size() - 1);
    const auto lo = std::size_t(index), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (index - lo);
}
} // namespace
const std::vector<std::string> &speakerFeatureNames() {
    static const auto names = [] {
        std::vector<std::string> n{"duration_s",         "active_fraction",  "periodic_fraction",
                                   "pitch_p10_hz",       "pitch_p50_hz",     "pitch_p90_hz",
                                   "pitch_sd_semitones", "periodicity_mean", "harmonicity_median_db",
                                   "centroid_mean_hz",   "centroid_sd_hz",   "flatness_mean",
                                   "flatness_sd",        "zcr_mean",         "zcr_sd",
                                   "flux_mean",          "flux_sd",          "level_mean_dbfs",
                                   "level_sd_db",        "clipped_fraction"};
        for (int c = 1; c <= 13; ++c) {
            n.push_back("mfcc" + std::to_string(c) + "_mean");
            n.push_back("mfcc" + std::to_string(c) + "_sd");
        }
        return n;
    }();
    return names;
}
std::vector<float> resampleForDescriptor(const std::vector<float> &input, int rate) {
    if (rate < 8000 || rate > 192000)
        throw std::invalid_argument("Unsupported sample rate");
    if (rate == descriptorSampleRate)
        return input;
    constexpr int phases = 512, taps = 64;
    std::vector<std::array<double, taps>> kernel(phases);
    const double cutoff = 0.46 * std::min(1.0, double(descriptorSampleRate) / rate);
    for (int phase = 0; phase < phases; ++phase) {
        double total = 0;
        for (int tap = 0; tap < taps; ++tap) {
            const double d = tap - (taps / 2 - 1) - double(phase) / phases;
            const double sinc = std::abs(d) < 1e-12 ? 2 * cutoff : std::sin(2 * pi * cutoff * d) / (pi * d);
            const double window = 0.5 + 0.5 * std::cos(pi * d / (taps / 2));
            kernel[std::size_t(phase)][std::size_t(tap)] = sinc * window;
            total += sinc * window;
        }
        for (auto &weight : kernel[std::size_t(phase)])
            weight /= total;
    }
    std::vector<float> out(std::size_t(std::ceil(double(input.size()) * descriptorSampleRate / rate)));
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double position = double(i) * rate / descriptorSampleRate;
        const auto center = std::ptrdiff_t(position);
        const auto phase = std::min(phases - 1, int((position - center) * phases));
        double value = 0;
        for (int tap = 0; tap < taps; ++tap) {
            const auto index = center + tap - (taps / 2 - 1);
            if (index >= 0 && index < std::ptrdiff_t(input.size())) {
                const auto x = input[std::size_t(index)];
                value += (std::isfinite(x) ? std::clamp(double(x), -1.0, 1.0) : 0) *
                         kernel[std::size_t(phase)][std::size_t(tap)];
            }
        }
        out[i] = float(value);
    }
    return out;
}
const std::vector<std::string> &extendedSpeakerFeatureNames() {
    static const auto names = [] {
        auto result = speakerFeatureNames();
        for (int formant = 1; formant <= 4; ++formant) {
            result.push_back("f" + std::to_string(formant) + "_median_hz");
            result.push_back("f" + std::to_string(formant) + "_iqr_hz");
        }
        for (int formant = 1; formant <= 4; ++formant)
            result.push_back("f" + std::to_string(formant) + "_bandwidth_median_hz");
        result.insert(result.end(),
                      {"formant_dispersion_median_hz", "formant_dispersion_sd_hz", "formant_valid_fraction",
                       "voiced_centroid_median_hz", "voiced_flatness_median"});
        for (int coefficient = 1; coefficient <= 13; ++coefficient)
            result.push_back("voiced_mfcc" + std::to_string(coefficient) + "_mean");
        return result;
    }();
    return names;
}
namespace {
std::vector<double> extractProfile(const std::vector<float> &input, int rate, bool extended) {
    const auto samples = resampleForDescriptor(input, rate);
    if (samples.size() < frameSize)
        throw std::invalid_argument("Audio is too short for a descriptor");
    Analyzer analyzer;
    std::vector<float> frame(frameSize);
    std::vector<double> pitch, semitones, periodicity, hnr, centroid, flatness, zcr, flux, level;
    std::array<std::vector<double>, 13> mfcc;
    std::array<std::vector<double>, 4> formants, bandwidths;
    std::array<std::vector<double>, 13> voicedMfcc;
    std::vector<double> dispersion, voicedCentroid, voicedFlatness;
    std::size_t frames = 0, active = 0;
    for (std::size_t start = 0; start + frameSize <= samples.size(); start += 512) {
        std::copy_n(samples.begin() + std::ptrdiff_t(start), frameSize, frame.begin());
        const auto f = analyzer.analyze(frame, descriptorSampleRate);
        ++frames;
        if (f.rmsDb < -50)
            continue;
        ++active;
        centroid.push_back(f.centroidHz);
        flatness.push_back(f.flatness);
        zcr.push_back(f.crossingRate);
        flux.push_back(f.flux);
        level.push_back(f.rmsDb);
        for (std::size_t c = 0; c < mfcc.size(); ++c)
            mfcc[c].push_back(f.mfcc[c]);
        if (f.pitchHz > 0 && f.periodicity >= 0.85) {
            pitch.push_back(f.pitchHz);
            semitones.push_back(12 * std::log2(f.pitchHz));
            periodicity.push_back(f.periodicity);
            if (extended) {
                voicedCentroid.push_back(f.centroidHz);
                voicedFlatness.push_back(f.flatness);
                for (std::size_t coefficient = 0; coefficient < voicedMfcc.size(); ++coefficient)
                    voicedMfcc[coefficient].push_back(f.mfcc[coefficient]);
                const auto resonances = estimateResonances(frame);
                if (resonances.valid) {
                    for (std::size_t formant = 0; formant < formants.size(); ++formant) {
                        formants[formant].push_back(resonances.frequencyHz[formant]);
                        bandwidths[formant].push_back(resonances.bandwidthHz[formant]);
                    }
                    dispersion.push_back((resonances.frequencyHz[3] - resonances.frequencyHz[0]) / 3);
                }
            }
        }
        if (f.harmonicityValid)
            hnr.push_back(f.harmonicityDb);
    }
    double clipped = 0;
    for (float x : input)
        if (std::isfinite(x) && std::abs(x) >= 0.999f)
            ++clipped;
    std::vector<double> values{double(input.size()) / rate,
                               double(active) / frames,
                               double(pitch.size()) / frames,
                               quantile(pitch, 0.1),
                               quantile(pitch, 0.5),
                               quantile(pitch, 0.9),
                               deviation(semitones),
                               mean(periodicity),
                               quantile(hnr, 0.5),
                               mean(centroid),
                               deviation(centroid),
                               mean(flatness),
                               deviation(flatness),
                               mean(zcr),
                               deviation(zcr),
                               mean(flux),
                               deviation(flux),
                               mean(level),
                               deviation(level),
                               clipped / std::max(std::size_t(1), input.size())};
    for (const auto &c : mfcc) {
        values.push_back(mean(c));
        values.push_back(deviation(c));
    }
    if (extended) {
        for (const auto &formant : formants) {
            values.push_back(quantile(formant, 0.5));
            values.push_back(quantile(formant, 0.75) - quantile(formant, 0.25));
        }
        for (const auto &bandwidth : bandwidths)
            values.push_back(quantile(bandwidth, 0.5));
        values.push_back(quantile(dispersion, 0.5));
        values.push_back(deviation(dispersion));
        values.push_back(double(dispersion.size()) / std::max(std::size_t(1), pitch.size()));
        values.push_back(quantile(voicedCentroid, 0.5));
        values.push_back(quantile(voicedFlatness, 0.5));
        for (const auto &coefficient : voicedMfcc)
            values.push_back(mean(coefficient));
    }
    return values;
}
} // namespace
std::vector<double> extractSpeakerFeatures(const std::vector<float> &input, int rate) {
    return extractProfile(input, rate, false);
}
std::vector<double> extractExtendedSpeakerFeatures(const std::vector<float> &input, int rate) {
    return extractProfile(input, rate, true);
}
} // namespace sona
