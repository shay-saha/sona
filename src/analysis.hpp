#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sona {
constexpr std::size_t frameSize = 2048;
constexpr std::size_t spectrumBands = 96;
constexpr double pi = 3.14159265358979323846;

struct Features {
    double rmsDb = -90;
    double pitchHz = 0;
    double periodicity = 0; // YIN periodicity score, not a calibrated probability.
    double centroidHz = 0;
    double flatness = 0;
    double flux = 0;
    double crossingRate = 0;
    double angularity = 0;    // Artistic mapping, not a trained Bouba/Kiki prediction.
    double harmonicityDb = 0; // Autocorrelation HNR estimate on periodic frames only.
    bool harmonicityValid = false;
    std::array<double, 13> mfcc{}; // C1–C13; no energy coefficient C0.
    bool active = false;
    std::array<float, spectrumBands> spectrum{};
    std::array<float, 256> waveform{};
};

class Analyzer {
  public:
    Features analyze(const std::vector<float> &samples, int sampleRate);
    void reset();

  private:
    std::array<double, frameSize / 2 + 1> previous_{};
    bool hasPrevious_ = false;
};

// Fully synthetic stimuli. Profiles are timbral contrasts, not human voices.
std::vector<float> synthesize(int profile, std::size_t offset, std::size_t count, int sampleRate);

struct WaveFile {
    std::vector<float> samples;
    int sampleRate = 0;
    std::string error;
    explicit operator bool() const {
        return error.empty() && !samples.empty();
    }
};
// RIFF/WAVE: integer PCM 16/24/32, IEEE float32; downmixes 1–8 channels.
WaveFile decodeWave(const std::vector<std::uint8_t> &bytes);
} // namespace sona
