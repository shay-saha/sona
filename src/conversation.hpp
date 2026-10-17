#pragma once
#include <QJsonObject>
#include <utility>
#include <vector>

namespace sona {
using TimeSpan = std::pair<double, double>;
struct SpeechSegment {
    int speaker = 0;
    double start = 0, end = 0;
};
struct ConversationSpeaker {
    int id = 0;
    double seconds = 0, share = 0;
    int turns = 0;
    bool active = false;
    double activity = 0;
};
struct TurnExchange {
    int from = 0, to = 0;
    double at = 0, gap = 0;
    bool hasResponseGap = false;
};
struct ConversationSummary {
    std::vector<ConversationSpeaker> speakers;
    std::vector<TurnExchange> exchanges;
    double duration = 0, speech = 0, overlap = 0, silence = 0, unobserved = 0;
};
struct DiarizationResult {
    std::vector<SpeechSegment> segments;
    std::vector<std::vector<float>> embeddings;
    double duration = 0;
    bool sessionStable = false;
};

std::vector<SpeechSegment> fuseSpeakerEvidence(const std::vector<SpeechSegment> &acoustic,
                                               const std::vector<SpeechSegment> &neural, double duration);

std::vector<SpeechSegment> normalizeSegments(const std::vector<SpeechSegment> &segments, double duration);
ConversationSummary summarizeConversation(const std::vector<SpeechSegment> &segments, double through,
                                          const std::vector<TimeSpan> &coverage = {});
QJsonObject conversationJson(const std::vector<SpeechSegment> &segments, double through,
                             const std::vector<TimeSpan> &coverage = {});
DiarizationResult parseDiarization(const QByteArray &json);

class ConversationHistory {
  public:
    void clear();
    void replace(const DiarizationResult &result, double offset, bool complete);
    const std::vector<SpeechSegment> &segments() const {
        return segments_;
    }
    double analyzedThrough() const {
        return analyzedThrough_;
    }
    const std::vector<TimeSpan> &coverage() const {
        return coverage_;
    }
    const std::vector<std::vector<float>> &voiceEmbeddings() const {
        return voices_;
    }

  private:
    std::vector<SpeechSegment> segments_;
    std::vector<std::vector<float>> voices_;
    std::vector<TimeSpan> coverage_;
    double analyzedThrough_ = 0;
};
} // namespace sona
