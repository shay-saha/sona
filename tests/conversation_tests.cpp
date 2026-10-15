#include "conversation.hpp"
#include <QJsonDocument>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
bool near(double a, double b) {
    return std::abs(a - b) < 1e-6;
}
} // namespace
int main() {
    try {
        using namespace sona;
        const std::vector<SpeechSegment> segments{{0, 0, 2},     {0, 1, 3}, {1, 2, 4},
                                                  {2, 2.5, 3.5}, {0, 5, 6}, {1, 6.4, 7.4}};
        const auto summary = summarizeConversation(segments, 8);
        check(near(summary.speech, 6) && near(summary.overlap, 1.5) && near(summary.silence, 2),
              "Speech/overlap must use wall-clock unions, including triple overlap");
        check(summary.speakers.size() == 3 && near(summary.speakers[0].seconds, 4) &&
                  near(summary.speakers[0].share, 0.5),
              "Duplicate speech must not inflate speaker time");
        check(summary.speakers[0].turns == 2 && summary.exchanges.size() == 4 &&
                  near(summary.exchanges[0].gap, -1) && near(summary.exchanges.back().gap, 0.4),
              "Exchanges must retain overlap and response gaps");
        check(
            summary.exchanges.back().hasResponseGap &&
                !summarizeConversation({{0, 0, 8}, {1, 1, 2}, {2, 3, 4}}, 9).exchanges.back().hasResponseGap,
            "Response gaps must exclude speech occupied by a third speaker");
        check(summarizeConversation({{0, 0, 2}, {1, 0, 2}, {2, 1, 3}}, 4).exchanges.empty(),
              "Simultaneous onsets must not create arbitrary directed exchanges");
        const auto partial = summarizeConversation(segments, 2.7);
        check(partial.speakers.size() == 3 && near(partial.overlap, 0.7) && partial.speakers[2].active &&
                  partial.speakers[0].turns == 1,
              "Playback summaries must exclude future evidence");
        const auto parsed = parseDiarization(QJsonDocument(conversationJson(segments, 8)).toJson());
        check(parsed.segments.size() == 5 && near(parsed.duration, 8), "Result JSON must round-trip");
        bool rejected = false;
        try {
            summarizeConversation({{0, 0, std::numeric_limits<double>::quiet_NaN()}}, 1);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "Nonfinite times must be rejected");
        ConversationHistory history;
        history.replace({{{0, 0, 3}, {1, 4, 7}}, {{1, 0}, {0, 1}}, 8}, 0, true);
        history.replace({{{1, 0, 1}, {0, 2, 5}, {1, 5.5, 7}}, {{0, 1}, {1, 0}}, 8}, 2, false);
        check(history.segments()[0].speaker == 0 && near(history.segments()[0].end, 3) &&
                  history.segments()[1].speaker == 1 && history.segments().back().speaker == 0,
              "Window cluster permutations must preserve session labels and replace shared evidence");
        history.replace({{{0, 0, 2}}, {{1, 0}}, 3}, 40, false);
        check(history.segments().back().speaker == 0, "Returning voices must match beyond the shared window");
        check(near(summarizeConversation(history.segments(), 43, history.coverage()).unobserved, 30),
              "Gaps between analyzed windows must not count as silence");
        history.clear();
        check(history.segments().empty() && history.analyzedThrough() == 0,
              "Source changes must clear session identity");
        std::cout << "Conversation accounting and session matching passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
