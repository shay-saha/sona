#include "conversation.hpp"
#include "local_diarizer.hpp"
#include "speaker_features.hpp"
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
#ifdef SONA_SHERPA
using sona::diarizeLocal;
using sona::diarizeRecording;
sona::DiarizationResult replayConversation(const std::vector<float> &samples, const QString &root,
                                           int speakers, float threshold, double window, double step,
                                           QJsonArray &timings) {
    sona::ConversationHistory history;
    for (double through = 5;; through += step) {
        const auto end = std::min(samples.size(), std::size_t(through * 16000));
        const auto start = end > std::size_t(window * 16000) ? end - std::size_t(window * 16000) : 0;
        const std::vector<float> chunk(samples.begin() + std::ptrdiff_t(start),
                                       samples.begin() + std::ptrdiff_t(end));
        QElapsedTimer timer;
        timer.start();
        const auto result = diarizeLocal(chunk, root, speakers, threshold);
        history.replace(result, double(start) / 16000, false);
        timings.append(timer.elapsed() / 1000.0);
        if (end == samples.size())
            break;
    }
    return {history.segments(), history.voiceEmbeddings(), double(samples.size()) / 16000};
}

#endif
} // namespace

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    try {
        const auto args = app.arguments();
        if (args.contains("--available")) {
#ifdef SONA_SHERPA
            return 0;
#else
            return 1;
#endif
        }
#ifndef SONA_SHERPA
        throw std::runtime_error(
            "Conversation runtime unavailable; fetch dependencies and rebuild with SONA_SHERPA_ROOT");
#else
        const auto option = [&args](const QString &key, const QString &fallback = {}) {
            const auto i = args.indexOf(key);
            return i >= 0 && i + 1 < args.size() ? args[i + 1] : fallback;
        };
        const int rate = option("--rate", "16000").toInt(), speakers = option("--speakers", "0").toInt();
        const float threshold = option("--cluster-threshold", "0.95").toFloat();
        if (!std::isfinite(threshold) || threshold <= 0 || threshold > 2)
            throw std::invalid_argument("Invalid clustering threshold");
        if (rate < 8000 || rate > 192000 || speakers < 0 || speakers > 32)
            throw std::invalid_argument("Invalid sample rate or speaker count");
        std::vector<float> samples;
        if (args.contains("--stdin")) {
            QByteArray bytes;
            char buffer[65536];
            while (std::cin.read(buffer, sizeof(buffer)) || std::cin.gcount()) {
                bytes.append(buffer, qsizetype(std::cin.gcount()));
                if (bytes.size() > 200 * 1024 * 1024)
                    throw std::invalid_argument("Oversize audio input");
            }
            if (bytes.size() % sizeof(float) || bytes.size() > 200 * 1024 * 1024)
                throw std::invalid_argument("Invalid or oversize audio input");
            samples.resize(std::size_t(bytes.size()) / sizeof(float));
            std::memcpy(samples.data(), bytes.constData(), std::size_t(bytes.size()));
        } else {
            QFile input(option("--wav"));
            if (!input.open(QIODevice::ReadOnly) || input.size() > 100 * 1024 * 1024)
                throw std::invalid_argument("Unreadable or oversize WAV");
            const auto bytes = input.readAll();
            auto wave = sona::decodeWave({bytes.begin(), bytes.end()});
            if (!wave)
                throw std::invalid_argument(wave.error);
            samples = sona::resampleForDescriptor(wave.samples, wave.sampleRate);
        }
        for (float s : samples)
            if (!std::isfinite(s) || std::abs(s) > 1.01f)
                throw std::invalid_argument("Audio samples must be finite and normalized");
        if (args.contains("--stdin"))
            samples = sona::resampleForDescriptor(samples, rate);
        if (samples.size() < 16000)
            throw std::invalid_argument("Collect at least one second of conversation audio");
        const double window = option("--rolling-window", "0").toDouble();
        const double step = option("--rolling-step", "2").toDouble();
        if (!std::isfinite(window) || !std::isfinite(step) || window < 0 || window > 60 ||
            (window > 0 && window < 5) || step < 0.5 || step > 10)
            throw std::invalid_argument("Invalid rolling replay settings");
        QJsonArray timings;
        const auto result = window > 0 ? replayConversation(samples, option("--models"), speakers, threshold,
                                                            window, step, timings)
                                       : diarizeRecording(samples, option("--models"), speakers, threshold);
        auto object = sona::conversationJson(result.segments, result.duration);
        QJsonArray embeddings;
        for (const auto &v : result.embeddings) {
            QJsonArray values;
            for (float value : v)
                values.append(value);
            embeddings.append(values);
        }
        object["embeddings"] = embeddings;
        if (!timings.isEmpty())
            object["window_wall_seconds"] = timings;
        object["backend"] = "sherpa-onnx 1.13.8 / pyannote segmentation 3.0 / TitaNet-S";
        object["configuration"] = QJsonObject{
            {"clustering_threshold", threshold}, {"expected_speakers", speakers}, {"sample_rate", 16000}};
        std::cout << QJsonDocument(object).toJson(QJsonDocument::Compact).constData() << '\n';
        return 0;
#endif
    } catch (const std::exception &e) {
        std::cout << QJsonDocument(QJsonObject{{"error", QString::fromUtf8(e.what())}})
                         .toJson(QJsonDocument::Compact)
                         .constData()
                  << '\n';
        return 1;
    }
}
