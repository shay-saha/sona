#include "conversation.hpp"
#include <QJsonArray>
#include <QJsonDocument>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace sona {
std::vector<SpeechSegment> normalizeSegments(const std::vector<SpeechSegment> &segments, double duration) {
    if (!std::isfinite(duration) || duration < 0)
        throw std::invalid_argument("Invalid conversation duration");
    std::vector<SpeechSegment> sorted;
    for (auto s : segments) {
        if (s.speaker < 0 || s.speaker >= 256 || !std::isfinite(s.start) || !std::isfinite(s.end))
            throw std::invalid_argument("Invalid speaker segment");
        s.start = std::clamp(s.start, 0.0, duration);
        s.end = std::clamp(s.end, 0.0, duration);
        if (s.end > s.start)
            sorted.push_back(s);
    }
    std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) {
        return std::tie(a.speaker, a.start, a.end) < std::tie(b.speaker, b.start, b.end);
    });
    std::vector<SpeechSegment> merged;
    for (auto s : sorted)
        if (!merged.empty() && merged.back().speaker == s.speaker && s.start <= merged.back().end + 1e-6)
            merged.back().end = std::max(merged.back().end, s.end);
        else
            merged.push_back(s);
    std::sort(merged.begin(), merged.end(),
              [](auto a, auto b) { return std::tie(a.start, a.speaker) < std::tie(b.start, b.speaker); });
    return merged;
}
ConversationSummary summarizeConversation(const std::vector<SpeechSegment> &segments, double through,
                                          const std::vector<TimeSpan> &coverage) {
    ConversationSummary result;
    result.duration = through;
    const auto normalized = normalizeSegments(segments, through);
    std::map<int, ConversationSpeaker> speakers;
    std::map<double, int> events;
    for (const auto &s : normalized) {
        auto &person = speakers[s.speaker];
        person.id = s.speaker;
        person.seconds += s.end - s.start;
        ++person.turns;
        person.active |= s.start <= through && s.end >= through - 0.08;
        ++events[s.start];
        --events[s.end];
    }
    double last = 0, total = 0;
    int active = 0;
    for (const auto &[at, delta] : events) {
        if (active > 0)
            result.speech += at - last;
        if (active > 1)
            result.overlap += at - last;
        active += delta;
        last = at;
    }
    double observed = through;
    if (!coverage.empty()) {
        std::vector<SpeechSegment> spans;
        for (auto [start, end] : coverage)
            spans.push_back({0, start, end});
        observed = 0;
        for (const auto &s : normalizeSegments(spans, through))
            observed += s.end - s.start;
    }
    result.unobserved = through - observed;
    result.silence = std::max(0.0, observed - result.speech);
    for (const auto &[id, person] : speakers) {
        (void)id;
        total += person.seconds;
    }
    for (auto &[id, person] : speakers) {
        (void)id;
        person.share = total > 0 ? person.seconds / total : 0;
        result.speakers.push_back(person);
    }
    // Exchanges follow consecutive turn onsets. Simultaneous onsets have no directed exchange.
    double precedingEnd = normalized.empty() ? 0 : normalized.front().end;
    for (std::size_t i = 1; i < normalized.size(); ++i) {
        const auto a = normalized[i - 1], b = normalized[i];
        const bool precedingTie = i > 1 && a.start - normalized[i - 2].start <= 0.001;
        const bool followingTie = i + 1 < normalized.size() && normalized[i + 1].start - b.start <= 0.001;
        if (a.speaker != b.speaker && b.start > a.start + 0.001 && !precedingTie && !followingTie) {
            const bool clearGap = b.start >= a.end && precedingEnd <= a.end + 1e-6;
            result.exchanges.push_back({a.speaker, b.speaker, b.start, b.start - a.end, clearGap});
        }
        precedingEnd = std::max(precedingEnd, b.end);
    }
    return result;
}
QJsonObject conversationJson(const std::vector<SpeechSegment> &segments, double through,
                             const std::vector<TimeSpan> &coverage) {
    const auto summary = summarizeConversation(segments, through, coverage);
    QJsonArray people, turns, exchanges, spans;
    if (coverage.empty())
        spans.append(QJsonArray{0, through});
    else
        for (auto [start, end] : coverage)
            spans.append(QJsonArray{start, end});
    for (const auto &p : summary.speakers)
        people.append(QJsonObject{{"speaker", p.id},
                                  {"speaking_seconds", p.seconds},
                                  {"speaking_share", p.share},
                                  {"turns", p.turns}});
    for (const auto &s : normalizeSegments(segments, through))
        turns.append(QJsonObject{{"speaker", s.speaker}, {"start", s.start}, {"end", s.end}});
    for (const auto &e : summary.exchanges)
        exchanges.append(QJsonObject{
            {"from", e.from},
            {"to", e.to},
            {"at", e.at},
            {"signed_gap_seconds", e.gap},
            {"response_gap_seconds", e.hasResponseGap ? QJsonValue(e.gap) : QJsonValue(QJsonValue::Null)}});
    return {{"schema_version", 1},
            {"duration_seconds", through},
            {"speech_seconds", summary.speech},
            {"overlap_seconds", summary.overlap},
            {"silence_seconds", summary.silence},
            {"unobserved_seconds", summary.unobserved},
            {"analyzed_intervals", spans},
            {"speakers", people},
            {"segments", turns},
            {"exchanges", exchanges},
            {"share_denominator", "sum of individual speaking seconds; overlap counts for each speaker"},
            {"turn_definition", "continuous detected activity for one speaker"},
            {"exchange_definition", "different speakers at consecutive non-simultaneous turn onsets"},
            {"gap_definition", "next onset minus preceding turn end; negative means overlap"},
            {"response_gap_definition", "nonnegative onset gap with no other speaker occupying the gap"}};
}
DiarizationResult parseDiarization(const QByteArray &json) {
    QJsonParseError error;
    const auto document = QJsonDocument::fromJson(json, &error);
    if (error.error != QJsonParseError::NoError || !document.isObject())
        throw std::invalid_argument("Invalid conversation result");
    const auto object = document.object();
    if (object.contains("error"))
        throw std::runtime_error(object.value("error").toString().toStdString());
    if (object.value("schema_version").toInt() != 1 || !object.value("segments").isArray() ||
        !object.value("duration_seconds").isDouble())
        throw std::invalid_argument("Unsupported conversation result");
    DiarizationResult result;
    result.duration = object.value("duration_seconds").toDouble();
    for (const auto &entry : object.value("segments").toArray()) {
        const auto s = entry.toObject();
        if (!s.value("speaker").isDouble() || !s.value("start").isDouble() || !s.value("end").isDouble())
            throw std::invalid_argument("Malformed speaker segment");
        result.segments.push_back(
            {s.value("speaker").toInt(-1), s.value("start").toDouble(), s.value("end").toDouble()});
    }
    result.segments = normalizeSegments(result.segments, result.duration);
    const auto embeddings = object.value("embeddings").toArray();
    if (embeddings.size() > 256)
        throw std::invalid_argument("Too many speaker embeddings");
    for (const auto &entry : embeddings) {
        std::vector<float> v;
        const auto values = entry.toArray();
        if (values.size() > 1024)
            throw std::invalid_argument("Oversize speaker embedding");
        for (const auto &value : values) {
            if (!value.isDouble() || !std::isfinite(value.toDouble()))
                throw std::invalid_argument("Invalid speaker embedding");
            v.push_back(float(value.toDouble()));
        }
        result.embeddings.push_back(std::move(v));
    }
    return result;
}
void ConversationHistory::clear() {
    segments_.clear();
    voices_.clear();
    coverage_.clear();
    analyzedThrough_ = 0;
}
void ConversationHistory::replace(const DiarizationResult &result, double offset, bool complete) {
    if (!std::isfinite(offset) || offset < 0)
        throw std::invalid_argument("Invalid conversation offset");
    if (complete)
        clear();
    const auto local = normalizeSegments(result.segments, result.duration);
    std::set<int> labels;
    for (const auto &s : local)
        labels.insert(s.speaker);
    struct Match {
        double score;
        int local, global;
    };
    std::vector<Match> candidates;
    for (int label : labels)
        for (std::size_t id = 0; id < voices_.size(); ++id) {
            double shared = 0, localTime = 0;
            for (const auto &s : local)
                if (s.speaker == label) {
                    localTime += std::max(0.0, std::min(s.end + offset, analyzedThrough_) - s.start - offset);
                    for (const auto &old : segments_)
                        if (old.speaker == int(id))
                            shared += std::max(0.0, std::min(s.end + offset, old.end) -
                                                        std::max(s.start + offset, old.start));
                }
            double cosine = -1;
            if (label < int(result.embeddings.size())) {
                const auto &a = result.embeddings[label], &b = voices_[id];
                if (!a.empty() && a.size() == b.size()) {
                    double dot = 0, aa = 0, bb = 0;
                    for (std::size_t k = 0; k < a.size(); ++k) {
                        dot += a[k] * b[k];
                        aa += a[k] * a[k];
                        bb += b[k] * b[k];
                    }
                    if (aa > 0 && bb > 0)
                        cosine = dot / std::sqrt(aa * bb);
                }
            }
            if (shared >= 0.5 && shared > localTime * 0.5)
                candidates.push_back({2 + shared / std::max(0.5, localTime), label, int(id)});
            else if (cosine >= 0.65)
                candidates.push_back({cosine, label, int(id)});
        }
    std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.score > b.score; });
    std::map<int, int> mapping;
    std::set<int> assigned;
    for (auto c : candidates)
        if (!mapping.count(c.local) && !assigned.count(c.global)) {
            mapping[c.local] = c.global;
            assigned.insert(c.global);
        }
    for (int label : labels) {
        if (!mapping.count(label)) {
            if (voices_.size() >= 256)
                throw std::runtime_error("Session speaker limit reached; reset conversation");
            mapping[label] = int(voices_.size());
            voices_.emplace_back();
        }
        if (label < int(result.embeddings.size()) && !result.embeddings[label].empty())
            voices_[mapping[label]] = result.embeddings[label];
    }
    std::vector<SpeechSegment> next;
    for (auto s : segments_)
        if (s.start < offset) {
            s.end = std::min(s.end, offset);
            next.push_back(s);
        }
    for (auto s : local)
        next.push_back({mapping.at(s.speaker), s.start + offset, s.end + offset});
    analyzedThrough_ = offset + result.duration;
    if (next.size() > 100000)
        throw std::runtime_error("Session turn limit reached; export and reset conversation");
    segments_ = normalizeSegments(next, analyzedThrough_);
    std::vector<SpeechSegment> spans{{0, offset, analyzedThrough_}};
    for (auto [start, end] : coverage_)
        spans.push_back({0, start, end});
    coverage_.clear();
    for (const auto &s : normalizeSegments(spans, analyzedThrough_))
        coverage_.emplace_back(s.start, s.end);
}
} // namespace sona
