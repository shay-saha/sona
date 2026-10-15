#include "audio_engine.hpp"
#include <QCoreApplication>
#include <QDataStream>
#include <QEventLoop>
#include <QFile>
#include <QTemporaryDir>
#include <QTimer>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
void runFor(int milliseconds) {
    QEventLoop loop;
    QTimer::singleShot(milliseconds, &loop, &QEventLoop::quit);
    loop.exec();
}
void fixture(const QString &path) {
    QFile file(path);
    check(file.open(QIODevice::WriteOnly), "Fixture must be writable");
    QDataStream out(&file);
    out.setByteOrder(QDataStream::LittleEndian);
    constexpr int frames = 24000;
    out.writeRawData("RIFF", 4);
    out << quint32(36 + frames * 2);
    out.writeRawData("WAVEfmt ", 8);
    out << quint32(16) << quint16(1) << quint16(1) << quint32(48000) << quint32(96000) << quint16(2)
        << quint16(16);
    out.writeRawData("data", 4);
    out << quint32(frames * 2);
    for (int i = 0; i < frames; ++i)
        out << qint16(std::sin(2 * sona::pi * 220 * i / 48000) * 16000);
}
} // namespace

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    try {
        sona::AudioEngine engine;
        int updates = 0;
        engine.onFrame = [&](const sona::Features &) { ++updates; };
        runFor(180);
        check(updates > 1 && engine.features().active, "The demo must produce analyzed frames");
        check(engine.recentAudio().size() > sona::frameSize,
              "Analysis must retain a recent in-memory audio snapshot");
        check(std::abs(engine.features().pitchHz - 164) < 3, "The warm demo must have the expected pitch");
        engine.setPaused(true);
        const auto revision = engine.sourceRevision();
        const auto elapsed = engine.elapsed();
        const int count = updates;
        const auto retained = engine.recentAudio().size();
        runFor(90);
        check(engine.elapsed() == elapsed && updates == count, "Pause must stop analysis and its timeline");
        check(engine.sourceRevision() == revision, "Pause must preserve source identity");
        check(engine.recentAudio().size() == retained, "Pause must stop the recent audio snapshot");
        const auto voiceRevision = engine.voiceRevision();
        const auto conversationSize = engine.conversationAudio().size();
        engine.beginVoiceProfile();
        check(engine.recentAudio().empty() && engine.conversationAudio().size() == conversationSize &&
                  engine.sourceRevision() == revision && engine.voiceRevision() > voiceRevision,
              "A new voice profile must exclude earlier speech without discarding conversation history");
        engine.setPaused(false);
        runFor(90);
        check(updates > count, "Resume must restart analysis");
        QTemporaryDir directory;
        check(directory.isValid(), "Temporary directory must exist");
        const auto path = directory.filePath("220hz.wav");
        fixture(path);
        QString error;
        check(engine.loadWave(path, error), "A valid file must load");
        check(engine.sourceRevision() > revision, "A new source must invalidate pending analyses");
        check(engine.profile() == -1 && engine.sampleRate() == 48000 && !engine.live(),
              "File source and sample rate must be preserved");
        runFor(150);
        check(std::abs(engine.features().pitchHz - 220) < 3, "Imported audio must reach the analyzer");
        engine.setPaused(true);
        const auto fileRevision = engine.sourceRevision();
        check(engine.seekFile(0.3, error) && std::abs(engine.filePosition() - 0.3) < 1e-6 &&
                  engine.sourceRevision() == fileRevision && engine.recentAudio().empty() && engine.paused(),
              "Seeking must preserve recording identity and pause state while clearing acoustic history");
        check(!engine.seekFile(std::nan(""), error), "Nonfinite seek positions must be rejected");
        engine.setPaused(false);
        engine.selectDemo(1);
        check(engine.sampleRate() == 24000 && engine.profile() == 1,
              "Switching to demo must reset its format");
        check(engine.recentAudio().empty(), "Changing source must clear the previous voice snapshot");
        check(engine.selectLoadedWave() && engine.sampleRate() == 48000,
              "Reselecting the file must restore its original rate");
        runFor(120);
        check(std::abs(engine.features().pitchHz - 220) < 3, "Reselected file must analyze correctly");
        check(!engine.loadWave(directory.filePath("missing.wav"), error),
              "Missing files must return an error");
        check(engine.profile() == -1 && engine.sampleRate() == 48000,
              "Failed imports must preserve the active source");
        std::cout << "Demo, pause, WAV import, and source transition checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
