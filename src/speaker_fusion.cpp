#include "conversation.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace sona {
std::vector<SpeechSegment> fuseSpeakerEvidence(const std::vector<SpeechSegment> &acoustic,
                                               const std::vector<SpeechSegment> &neural, double duration) {
    const auto old = normalizeSegments(acoustic, duration);
    const auto anchors = normalizeSegments(neural, duration);
    if (old.empty() || anchors.empty())
        return {};
    if (duration > 3600)
        throw std::invalid_argument("Fuse conversation evidence in windows of at most one hour");
    const auto frames = std::size_t(std::llround(duration * 100));
    std::map<int, std::size_t> columns;
    for (const auto &segment : old)
        if (!columns.count(segment.speaker))
            columns[segment.speaker] = columns.size();
    std::vector<std::uint8_t> acousticActivity(frames * columns.size()), neuralActivity(frames);
    const auto boundary = [frames](double seconds) {
        return std::size_t(std::clamp(std::ceil(seconds * 100 - 0.5), 0.0, double(frames)));
    };
    for (const auto &segment : old)
        for (auto frame = boundary(segment.start); frame < boundary(segment.end); ++frame)
            acousticActivity[frame * columns.size() + columns.at(segment.speaker)] = 1;
    for (const auto &segment : anchors) {
        if (segment.speaker >= 8)
            throw std::invalid_argument("Neural evidence exceeds eight speaker channels");
        for (auto frame = boundary(segment.start); frame < boundary(segment.end); ++frame)
            neuralActivity[frame] |= std::uint8_t(1U << segment.speaker);
    }
    std::vector<std::array<std::size_t, 8>> shared(columns.size());
    for (std::size_t frame = 0; frame < frames; ++frame)
        for (std::size_t cluster = 0; cluster < columns.size(); ++cluster)
            if (acousticActivity[frame * columns.size() + cluster])
                for (int voice = 0; voice < 8; ++voice)
                    if (neuralActivity[frame] & (1U << voice))
                        ++shared[cluster][voice];
    std::vector<std::uint8_t> mapping(columns.size());
    for (std::size_t cluster = 0; cluster < columns.size(); ++cluster) {
        const auto best = std::max_element(shared[cluster].begin(), shared[cluster].end());
        if (*best >= 10)
            mapping[cluster] = std::uint8_t(1U << std::distance(shared[cluster].begin(), best));
    }
    std::array<double, 8> starts;
    starts.fill(-1);
    std::vector<SpeechSegment> result;
    for (std::size_t frame = 0; frame <= frames; ++frame) {
        bool speech = false;
        std::uint8_t assigned = 0;
        if (frame < frames) {
            for (std::size_t cluster = 0; cluster < columns.size(); ++cluster)
                if (acousticActivity[frame * columns.size() + cluster]) {
                    speech = true;
                    assigned |= mapping[cluster];
                }
            if (speech && neuralActivity[frame])
                assigned = neuralActivity[frame];
        }
        for (int voice = 0; voice < 8; ++voice) {
            const bool active = (assigned & (1U << voice)) != 0;
            if (active && starts[voice] < 0)
                starts[voice] = frame / 100.0;
            else if (!active && starts[voice] >= 0) {
                result.push_back({voice, starts[voice], std::min(duration, frame / 100.0)});
                starts[voice] = -1;
            }
        }
    }
    return normalizeSegments(result, duration);
}
} // namespace sona
