#pragma once
#include "analysis.hpp"
#include <string>
#include <vector>

namespace sona {
constexpr int descriptorSampleRate = 16000;
namespace speakerFeature {
enum Index : std::size_t {
    DurationSeconds,
    ActiveFraction,
    PeriodicFraction,
    PitchP10Hz,
    MedianPitchHz,
    PitchP90Hz,
    PitchDeviationSemitones,
    MeanPeriodicity,
    MedianHarmonicityDb,
    MeanCentroidHz,
    CentroidDeviationHz,
    MeanFlatness,
    FlatnessDeviation,
    MeanZeroCrossingRate,
    ZeroCrossingDeviation,
    MeanFlux,
    FluxDeviation,
    MeanLevelDbfs,
    LevelDeviationDb,
    ClippedFraction,
    FirstMfccMean
};
}
const std::vector<std::string> &speakerFeatureNames();
std::vector<float> resampleForDescriptor(const std::vector<float> &samples, int sampleRate);
// Fixed schema; acoustic descriptors, without any personal-characteristic predictions.
std::vector<double> extractSpeakerFeatures(const std::vector<float> &samples, int sampleRate);
} // namespace sona
