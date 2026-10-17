#include "conversation_controller.hpp"
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QTemporaryDir>
#include <QTimer>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
bool waitFor(const std::function<bool()> &predicate, int milliseconds) {
    QElapsedTimer clock;
    clock.start();
    while (!predicate() && clock.elapsed() < milliseconds) {
        QEventLoop loop;
        QTimer::singleShot(10, &loop, &QEventLoop::quit);
        loop.exec();
    }
    return predicate();
}
} // namespace
int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    try {
        QTemporaryDir directory;
        QFile marker(directory.filePath("nemo_en_titanet_small.onnx"));
        check(marker.open(QIODevice::WriteOnly), "Model marker must be writable");
        marker.close();
        QFile worker(directory.filePath("worker.py"));
        check(worker.open(QIODevice::WriteOnly), "Worker fixture must be writable");
        worker.write(R"(#!/usr/bin/env python3
import json, sys, time
samples = len(sys.stdin.buffer.read()) // 4
time.sleep(0.1)
print(json.dumps({'schema_version': 1, 'duration_seconds': samples / 16000,
                  'segments': [{'speaker': 0, 'start': 0, 'end': 1.5},
                               {'speaker': 1, 'start': 1, 'end': 2}]}), flush=True)
)");
        worker.close();
        worker.setPermissions(QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
        sona::ConversationController controller(directory.path(), nullptr, worker.fileName());
        const std::vector<float> samples(16000 * 3, 0);
        controller.analyze(samples, 16000, 0, true);
        check(controller.busy(), "Background analysis must start asynchronously");
        check(waitFor([&] { return !controller.busy(); }, 3000), "Worker must finish");
        check(controller.history().analyzedThrough() == 3 && controller.history().segments().size() == 2,
              "The whole audio payload and returned speaker segments must reach the controller");
        controller.analyze(samples, 16000, 10, false);
        controller.reset();
        check(waitFor([&] { return !controller.busy(); }, 3000) && controller.history().segments().empty(),
              "Source changes must cancel the old worker and clear its result");
        controller.analyze(samples, 16000, 0, true);
        check(waitFor([&] { return !controller.busy(); }, 3000) &&
                  controller.history().analyzedThrough() == 3,
              "A new source must be analyzable after cancellation");
        sona::ConversationController missing(directory.path(), nullptr, directory.filePath("missing"));
        missing.analyze(samples, 16000, 0, true);
        check(waitFor([&] { return !missing.busy(); }, 3000) && missing.status().contains("could not start"),
              "Missing workers must report an actionable error without hanging");
        QFile neuralMarker(directory.filePath("Nemotron-3-Diarization.q8_0.gguf"));
        check(neuralMarker.open(QIODevice::WriteOnly), "Neural marker must be writable");
        neuralMarker.close();
        QFile streamingWorker(directory.filePath("stream.py"));
        check(streamingWorker.open(QIODevice::WriteOnly), "Persistent worker fixture must be writable");
        streamingWorker.write(R"(#!/usr/bin/env python3
import json, struct, sys
if '--available' in sys.argv:
    sys.exit(0)
received = 0
while header := sys.stdin.buffer.read(4):
    count, = struct.unpack('<I', header)
    payload = sys.stdin.buffer.read(count * 4)
    assert len(payload) == count * 4
    received += count
    through = received / 16000
    print(json.dumps({'schema_version': 1, 'identity_space': 'session',
                      'offset_seconds': 0, 'duration_seconds': through,
                      'segments': [{'speaker': 2, 'start': 0, 'end': through}],
                      'activity_scores': [0.01, 0.02, 0.9]}), flush=True)
)");
        streamingWorker.close();
        streamingWorker.setPermissions(QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
        sona::ConversationController streaming(directory.path(), nullptr, worker.fileName(),
                                               streamingWorker.fileName());
        check(streaming.usesStreaming(), "Available neural workers must use the persistent protocol");
        const std::vector<float> oneSecond(16000, 0), twoSeconds(32000, 0);
        streaming.analyze(oneSecond, 16000, 0, false);
        check(waitFor([&] { return !streaming.busy(); }, 3000) && streaming.history().analyzedThrough() == 1,
              "Persistent workers must acknowledge an initial audio chunk");
        streaming.analyze(twoSeconds, 16000, 0, false);
        check(waitFor([&] { return !streaming.busy(); }, 3000) && streaming.history().analyzedThrough() == 2,
              "Overlapping capture buffers must send each sample only once");
        streaming.analyze(twoSeconds, 16000, 1, false);
        check(waitFor([&] { return !streaming.busy(); }, 3000) &&
                  streaming.history().analyzedThrough() == 3 && streaming.history().segments().size() == 1 &&
                  streaming.history().segments().front().speaker == 2 &&
                  streaming.activityScores().size() == 3 && streaming.activityScores()[2] > 0.8,
              "Persistent arrival identities and activity must survive successive capture windows");
        streaming.analyze(oneSecond, 16000, 20, false);
        check(waitFor([&] { return !streaming.busy(); }, 3000) &&
                  streaming.history().analyzedThrough() == 21 &&
                  streaming.history().segments().front().start == 20,
              "Audio gaps must restart tracking and preserve the source clock");
        streaming.reset();
        check(streaming.history().segments().empty() && streaming.activityScores().empty(),
              "Source reset must discard neural identity and activity evidence");
        std::cout << "Conversation transfer, cancellation and recovery passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
