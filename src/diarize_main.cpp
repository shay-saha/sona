#include "conversation.hpp"
#include "speaker_features.hpp"
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
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
#ifdef SONA_SHERPA
#include <sherpa-onnx/c-api/c-api.h>
#endif

namespace {
#ifdef SONA_SHERPA
void verify(const QString &path, const char *expected) {
    QFile file(path);
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!file.open(QIODevice::ReadOnly) || !hash.addData(&file) || hash.result().toHex() != expected)
        throw std::runtime_error(
            "Conversation model missing or checksum mismatch; run research/scripts/fetch_conversation.py");
}
sona::DiarizationResult diarize(const std::vector<float> &samples, const QString &root, int speakers,
                                float threshold, double progressBegin = 0, double progressSpan = 1) {
    const auto segmentation = QDir(root).filePath("sherpa-onnx-pyannote-segmentation-3-0/model.onnx");
    const auto embedding = QDir(root).filePath("nemo_en_titanet_small.onnx");
    verify(segmentation, "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079");
    verify(embedding, "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e");
    const auto segPath = QFile::encodeName(segmentation), embPath = QFile::encodeName(embedding);
    SherpaOnnxOfflineSpeakerDiarizationConfig config{};
    config.segmentation.pyannote.model = segPath.constData();
    config.segmentation.pyannote.window_shift_ratio = 0.1f;
    config.segmentation.num_threads = 2;
    config.segmentation.provider = "cpu";
    config.embedding.model = embPath.constData();
    config.embedding.num_threads = 2;
    config.embedding.provider = "cpu";
    config.clustering.num_clusters = speakers;
    config.clustering.threshold = threshold;
    config.min_duration_on = 0.2f;
    config.min_duration_off = 0.1f;
    const auto handle = std::unique_ptr<const SherpaOnnxOfflineSpeakerDiarization,
                                        decltype(&SherpaOnnxDestroyOfflineSpeakerDiarization)>(
        SherpaOnnxCreateOfflineSpeakerDiarization(&config), SherpaOnnxDestroyOfflineSpeakerDiarization);
    if (!handle)
        throw std::runtime_error("Could not initialize conversation models");
    struct Progress {
        double begin, span;
    } progress{progressBegin, progressSpan};
    const auto result = std::unique_ptr<const SherpaOnnxOfflineSpeakerDiarizationResult,
                                        decltype(&SherpaOnnxOfflineSpeakerDiarizationDestroyResult)>(
        SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
            handle.get(), samples.data(), int(samples.size()),
            [](int32_t done, int32_t total, void *data) -> int32_t {
                const auto &p = *static_cast<const Progress *>(data);
                if (total > 0 && (done % 10 == 0 || done == total))
                    std::cerr << "SONA_PROGRESS " << int((p.begin + p.span * done / total) * 100) << '\n';
                return 0;
            },
            &progress),
        SherpaOnnxOfflineSpeakerDiarizationDestroyResult);
    if (!result)
        throw std::runtime_error("Conversation analysis failed");
    const int count = SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result.get());
    const auto segments = std::unique_ptr<const SherpaOnnxOfflineSpeakerDiarizationSegment,
                                          decltype(&SherpaOnnxOfflineSpeakerDiarizationDestroySegment)>(
        SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result.get()),
        SherpaOnnxOfflineSpeakerDiarizationDestroySegment);
    sona::DiarizationResult output;
    output.duration = samples.size() / 16000.0;
    int labelCount = 0;
    for (int i = 0; i < count; ++i) {
        const auto s = segments.get()[i];
        if (s.speaker < 0 || s.speaker >= 256)
            throw std::runtime_error("Conversation speaker limit exceeded");
        output.segments.push_back({s.speaker, s.start, s.end});
        labelCount = std::max(labelCount, s.speaker + 1);
    }
    output.segments = sona::normalizeSegments(output.segments, output.duration);
    output.embeddings.resize(labelCount);
    // Only isolated speech contributes to the session voice centroids.
    const auto extractor = std::unique_ptr<const SherpaOnnxSpeakerEmbeddingExtractor,
                                           decltype(&SherpaOnnxDestroySpeakerEmbeddingExtractor)>(
        SherpaOnnxCreateSpeakerEmbeddingExtractor(&config.embedding),
        SherpaOnnxDestroySpeakerEmbeddingExtractor);
    if (!extractor)
        throw std::runtime_error("Could not initialize session voice matching");
    for (int label = 0; label < labelCount; ++label) {
        std::vector<float> isolated;
        for (const auto &s : output.segments) {
            if (s.speaker != label)
                continue;
            std::vector<std::pair<double, double>> spans{{s.start, s.end}};
            for (const auto &other : output.segments) {
                if (other.speaker == label)
                    continue;
                std::vector<std::pair<double, double>> next;
                for (const auto &[a, b] : spans) {
                    if (other.end <= a || other.start >= b)
                        next.emplace_back(a, b);
                    else {
                        if (other.start > a)
                            next.emplace_back(a, other.start);
                        if (other.end < b)
                            next.emplace_back(other.end, b);
                    }
                }
                spans = std::move(next);
            }
            for (const auto &[a, b] : spans) {
                const auto first = std::min(samples.size(), std::size_t(a * 16000));
                const auto end = std::min(samples.size(), std::size_t(b * 16000));
                const auto n = std::min(end - first, std::size_t(8 * 16000) - isolated.size());
                isolated.insert(isolated.end(), samples.begin() + std::ptrdiff_t(first),
                                samples.begin() + std::ptrdiff_t(first + n));
                if (isolated.size() >= 8 * 16000)
                    break;
            }
            if (isolated.size() >= 8 * 16000)
                break;
        }
        if (isolated.size() < 16000)
            continue;
        const auto stream =
            std::unique_ptr<const SherpaOnnxOnlineStream, decltype(&SherpaOnnxDestroyOnlineStream)>(
                SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor.get()),
                SherpaOnnxDestroyOnlineStream);
        SherpaOnnxOnlineStreamAcceptWaveform(stream.get(), 16000, isolated.data(), int(isolated.size()));
        SherpaOnnxOnlineStreamInputFinished(stream.get());
        if (!SherpaOnnxSpeakerEmbeddingExtractorIsReady(extractor.get(), stream.get()))
            continue;
        const auto vector =
            std::unique_ptr<const float, decltype(&SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding)>(
                SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor.get(), stream.get()),
                SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding);
        if (vector) {
            const int dimensions = SherpaOnnxSpeakerEmbeddingExtractorDim(extractor.get());
            output.embeddings[label] = {vector.get(), vector.get() + dimensions};
        }
    }
    return output;
}
sona::DiarizationResult diarizeRecording(const std::vector<float> &samples, const QString &root, int speakers,
                                         float threshold) {
    if (samples.size() <= 300 * 16000)
        return diarize(samples, root, speakers, threshold);
    sona::ConversationHistory history;
    std::size_t start = 0, processed = 0;
    while (processed < samples.size()) {
        const auto end = std::min(samples.size(), start + 180 * 16000);
        const std::vector<float> chunk(samples.begin() + std::ptrdiff_t(start),
                                       samples.begin() + std::ptrdiff_t(end));
        const auto result = diarize(chunk, root, speakers, threshold, double(processed) / samples.size(),
                                    double(end - processed) / samples.size());
        history.replace(result, double(start) / 16000, start == 0);
        processed = end;
        if (end < samples.size())
            start = end - 30 * 16000;
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
        const auto result = diarizeRecording(samples, option("--models"), speakers, threshold);
        auto object = sona::conversationJson(result.segments, result.duration);
        QJsonArray embeddings;
        for (const auto &v : result.embeddings) {
            QJsonArray values;
            for (float value : v)
                values.append(value);
            embeddings.append(values);
        }
        object["embeddings"] = embeddings;
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
