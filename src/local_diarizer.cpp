#include "local_diarizer.hpp"
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#ifdef SONA_SHERPA
#include <sherpa-onnx/c-api/c-api.h>
#endif

namespace sona {
#ifdef SONA_SHERPA
namespace {
void verify(const QString &path, const char *expected) {
    QFile file(path);
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!file.open(QIODevice::ReadOnly) || !hash.addData(&file) || hash.result().toHex() != expected)
        throw std::runtime_error(
            "Conversation model missing or checksum mismatch; run research/scripts/fetch_conversation.py");
}
} // namespace
DiarizationResult diarizeLocal(const std::vector<float> &samples, const QString &root, int speakers,
                               float threshold, double progressBegin, double progressSpan,
                               bool collectEmbeddings) {
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
    if (!collectEmbeddings)
        return output;
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
DiarizationResult diarizeRecording(const std::vector<float> &samples, const QString &root, int speakers,
                                   float threshold, bool collectEmbeddings, double progressBegin,
                                   double progressSpan) {
    if (samples.size() <= 300 * 16000)
        return diarizeLocal(samples, root, speakers, threshold, progressBegin, progressSpan,
                            collectEmbeddings);
    sona::ConversationHistory history;
    std::size_t start = 0, processed = 0;
    while (processed < samples.size()) {
        const auto end = std::min(samples.size(), start + 180 * 16000);
        const std::vector<float> chunk(samples.begin() + std::ptrdiff_t(start),
                                       samples.begin() + std::ptrdiff_t(end));
        const auto result =
            diarizeLocal(chunk, root, speakers, threshold,
                         progressBegin + progressSpan * double(processed) / samples.size(),
                         progressSpan * double(end - processed) / samples.size(), collectEmbeddings);
        history.replace(result, double(start) / 16000, start == 0);
        processed = end;
        if (end < samples.size())
            start = end - 30 * 16000;
    }
    return {history.segments(), history.voiceEmbeddings(), double(samples.size()) / 16000};
}
#else
DiarizationResult diarizeLocal(const std::vector<float> &, const QString &, int, float, double, double,
                               bool) {
    throw std::runtime_error("Acoustic conversation runtime unavailable");
}
DiarizationResult diarizeRecording(const std::vector<float> &, const QString &, int, float, bool, double,
                                   double) {
    throw std::runtime_error("Acoustic conversation runtime unavailable");
}
#endif
} // namespace sona
