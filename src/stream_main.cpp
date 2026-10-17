#include "conversation.hpp"
#include "local_diarizer.hpp"
#include "speaker_features.hpp"
#include "stream_protocol.hpp"
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#ifdef SONA_NEMOTRON
#include <nemo_speech/diar.h>
#endif

namespace {
#ifdef SONA_NEMOTRON
void check(nemo_speech_asr_status status) {
    if (status != NEMO_SPEECH_ASR_OK)
        throw std::runtime_error(nemo_speech_asr_last_error());
}
using Model = std::unique_ptr<nemo_speech_diar_model, decltype(&nemo_speech_diar_destroy)>;
using Stream = std::unique_ptr<nemo_speech_diar_stream, decltype(&nemo_speech_diar_stream_close)>;

Model loadModel(const QString &root, const QString &preset, int &gpu, bool fallback) {
    const auto path = QDir(root).filePath("Nemotron-3-Diarization.q8_0.gguf");
    QFile file(path);
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!file.open(QIODevice::ReadOnly) || !hash.addData(&file) ||
        hash.result().toHex() != "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1")
        throw std::runtime_error("Neural conversation model missing or checksum mismatch");
    const auto modelPath = QFile::encodeName(path), geometry = preset.toUtf8();
    nemo_speech_diar_model_config config{};
    config.size = sizeof(config);
    config.model_path = modelPath.constData();
    config.gpu = gpu;
    config.preset = geometry.constData();
    config.left_context_frames = -1;
    nemo_speech_diar_model *model = nullptr;
    auto status = nemo_speech_diar_create(&config, &model);
    if (status != NEMO_SPEECH_ASR_OK && gpu >= 0 && fallback) {
        config.gpu = gpu = -1;
        status = nemo_speech_diar_create(&config, &model);
    }
    check(status);
    return {model, nemo_speech_diar_destroy};
}

std::vector<sona::SpeechSegment> neuralSegments(nemo_speech_diar_stream *stream, double duration) {
    nemo_speech_diar_segmentation_config config{};
    config.size = sizeof(config);
    config.onset = 0.5f;
    config.offset = 0.3f;
    config.pad_onset_sec = config.pad_offset_sec = 0.03;
    config.min_gap_sec = config.min_duration_sec = 0.1;
    std::size_t count = 0;
    check(nemo_speech_diar_segments(stream, &config, nullptr, 0, &count));
    std::vector<nemo_speech_diar_segment> raw(count);
    if (count)
        check(nemo_speech_diar_segments(stream, &config, raw.data(), raw.size(), &count));
    std::vector<sona::SpeechSegment> segments;
    for (const auto &segment : raw)
        segments.push_back({segment.speaker - 1, segment.start_time, segment.end_time});
    return sona::normalizeSegments(segments, duration);
}

QJsonObject snapshot(const std::vector<sona::SpeechSegment> &segments, nemo_speech_diar_model *model,
                     double duration) {
    auto result = sona::conversationJson(segments, duration);
    result["identity_space"] = "session";
    result["backend"] = "Nemotron 3 Diarization / Q8_0 / NeMo-Speech.cpp";
    result["speaker_capacity"] = nemo_speech_diar_num_speakers(model);
    result["frame_seconds"] = nemo_speech_diar_seconds_per_frame(model);
    return result;
}

std::vector<sona::SpeechSegment> localSegments(const std::vector<sona::SpeechSegment> &segments, double begin,
                                               double through) {
    std::vector<sona::SpeechSegment> result;
    for (const auto &segment : segments)
        if (segment.end > begin && segment.start < through)
            result.push_back({segment.speaker, std::max(0.0, segment.start - begin),
                              std::min(through, segment.end) - begin});
    return result;
}

void activityScores(QJsonObject &object, nemo_speech_diar_stream *stream, nemo_speech_diar_model *model) {
    const auto first = nemo_speech_diar_frame_probs_start(stream);
    const auto frames = nemo_speech_diar_frame_count(stream) - first;
    const int voices = nemo_speech_diar_num_speakers(model);
    if (frames <= 0)
        return;
    std::vector<float> probabilities(std::size_t(frames) * voices);
    check(nemo_speech_diar_frame_probs(stream, probabilities.data(), probabilities.size()));
    const auto start = std::max(std::int64_t(0), frames - 10);
    QJsonArray scores;
    for (int voice = 0; voice < voices; ++voice) {
        double sum = 0;
        for (auto frame = start; frame < frames; ++frame)
            sum += probabilities[std::size_t(frame) * voices + voice];
        scores.append(sum / (frames - start));
    }
    object["activity_scores"] = scores;
}

void serve(nemo_speech_diar_stream *stream, nemo_speech_diar_model *model, const QString &root, int rate,
           bool fused, int gpu) {
    if (rate > 96000)
        throw std::invalid_argument("Streaming sample rates above 96 kHz are unsupported");
    std::vector<float> recent, chunk;
    std::uint64_t received = 0;
    double lastAcoustic = 0, lastLaunched = 0, lastPublished = 0;
    struct AcousticWindow {
        double offset, through;
        std::vector<sona::SpeechSegment> segments;
    };
    std::future<AcousticWindow> pending;
    for (;;) {
        const bool final = !sona::readAudioChunk(std::cin, chunk) || chunk.empty();
        if (final)
            check(nemo_speech_diar_stream_finish(stream));
        else {
            received += chunk.size();
            recent.insert(recent.end(), chunk.begin(), chunk.end());
            if (recent.size() > std::size_t(rate) * 30)
                recent.erase(recent.begin(), recent.end() - std::ptrdiff_t(rate * 30));
            check(nemo_speech_diar_stream_push_f32(stream, chunk.data(), chunk.size(), rate));
        }
        const double captured = double(received) / rate;
        const double through = final ? captured
                                     : std::min(captured, nemo_speech_diar_frame_count(stream) *
                                                              nemo_speech_diar_seconds_per_frame(model));
        const auto neural = neuralSegments(stream, through);
        double offset = std::max(lastAcoustic, std::max(0.0, lastPublished - 0.3));
        std::vector<sona::SpeechSegment> segments;
        if (final && pending.valid())
            pending.wait();
        if (fused && through >= 5 && (!pending.valid() || final) && (through - lastLaunched >= 5 || final)) {
            if (pending.valid())
                pending.get();
            const auto ringBegin = received - recent.size();
            const auto end = std::clamp(std::uint64_t(through * rate), ringBegin, received);
            const auto begin =
                std::max(ringBegin, end > std::uint64_t(rate) * 30 ? end - std::uint64_t(rate) * 30 : 0);
            const double start = double(begin) / rate;
            const std::vector<float> window(recent.begin() + std::ptrdiff_t(begin - ringBegin),
                                            recent.begin() + std::ptrdiff_t(end - ringBegin));
            pending = std::async(std::launch::async, [window, rate, root, start, through] {
                const auto audio = sona::resampleForDescriptor(window, rate);
                const auto acoustic = sona::diarizeLocal(audio, root, 0, 0.95f, 0, 1, false);
                return AcousticWindow{start, through, acoustic.segments};
            });
            lastLaunched = through;
        }
        if (pending.valid() &&
            (final || pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
            const auto acoustic = pending.get();
            offset = acoustic.offset;
            segments =
                sona::fuseSpeakerEvidence(acoustic.segments, localSegments(neural, offset, acoustic.through),
                                          acoustic.through - offset);
            for (const auto &segment : localSegments(neural, acoustic.through, through))
                segments.push_back({segment.speaker, segment.start + acoustic.through - offset,
                                    segment.end + acoustic.through - offset});
            lastAcoustic = acoustic.through;
        } else
            segments = localSegments(neural, offset, through);
        auto result = snapshot(segments, model, std::max(0.0, through - offset));
        result["offset_seconds"] = offset;
        result["received_seconds"] = captured;
        result["final"] = final;
        result["device"] = gpu >= 0 ? "gpu" : "cpu";
        result["acoustic_through_seconds"] = lastAcoustic;
        activityScores(result, stream, model);
        std::cout << QJsonDocument(result).toJson(QJsonDocument::Compact).constData() << std::endl;
        lastPublished = through;
        if (final)
            break;
    }
}
#endif
} // namespace

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    const auto args = app.arguments();
    if (args.contains("--available")) {
#if defined(SONA_NEMOTRON) && defined(SONA_SHERPA)
        return 0;
#else
        return 1;
#endif
    }
    try {
#ifndef SONA_NEMOTRON
        throw std::runtime_error("Neural conversation runtime unavailable; fetch dependencies and rebuild");
#else
        const auto option = [&args](const QString &key, const QString &fallback = {}) {
            const auto i = args.indexOf(key);
            return i >= 0 && i + 1 < args.size() ? args[i + 1] : fallback;
        };
        const auto preset = option("--preset", "v3-streaming");
        if (preset != "v3-streaming" && preset != "v3-offline")
            throw std::invalid_argument("Unsupported neural conversation preset");
        bool valid = false;
        const int rate = option("--rate", "16000").toInt(&valid);
        if (!valid || rate < 8000 || rate > 192000)
            throw std::invalid_argument("Unsupported sample rate");
        int gpu = -1;
#ifdef SONA_NEMOTRON_CUDA
        gpu = 0;
#endif
        if (args.contains("--cpu"))
            gpu = -1;
        if (args.contains("--gpu"))
            gpu = 0;
        auto model = loadModel(option("--models"), preset, gpu, !args.contains("--gpu"));
        nemo_speech_diar_stream *handle = nullptr;
        check(nemo_speech_diar_stream_open(model.get(), &handle));
        Stream stream(handle, nemo_speech_diar_stream_close);
        const bool fused = !args.contains("--neural-only");
        if (args.contains("--stream")) {
            serve(stream.get(), model.get(), option("--models"), rate, fused, gpu);
            return 0;
        }
        std::vector<float> samples;
        if (args.contains("--stdin")) {
            QByteArray bytes;
            char buffer[65536];
            while (std::cin.read(buffer, sizeof(buffer)) || std::cin.gcount()) {
                bytes.append(buffer, qsizetype(std::cin.gcount()));
                if (bytes.size() > 200 * 1024 * 1024)
                    throw std::invalid_argument("Oversize audio input");
            }
            if (bytes.size() % sizeof(float))
                throw std::invalid_argument("Invalid float audio input");
            samples.resize(std::size_t(bytes.size()) / sizeof(float));
            std::memcpy(samples.data(), bytes.constData(), std::size_t(bytes.size()));
            samples = sona::resampleForDescriptor(samples, rate);
        } else {
            QFile file(option("--wav"));
            if (!file.open(QIODevice::ReadOnly) || file.size() > 100 * 1024 * 1024)
                throw std::invalid_argument("Unreadable or oversize WAV");
            const auto bytes = file.readAll();
            const auto wave = sona::decodeWave({bytes.begin(), bytes.end()});
            if (!wave)
                throw std::invalid_argument(wave.error);
            samples = sona::resampleForDescriptor(wave.samples, wave.sampleRate);
        }
        if (samples.empty())
            throw std::invalid_argument("Empty conversation audio");
        for (float sample : samples)
            if (!std::isfinite(sample) || std::abs(sample) > 1.01f)
                throw std::invalid_argument("Audio samples must be finite and normalized");
        QJsonArray timings;
        for (std::size_t offset = 0; offset < samples.size(); offset += 16000) {
            const auto count = std::min(std::size_t(16000), samples.size() - offset);
            QElapsedTimer timer;
            timer.start();
            check(nemo_speech_diar_stream_push_f32(stream.get(), samples.data() + offset, count, 16000));
            timings.append(timer.elapsed() / 1000.0);
            std::cerr << "SONA_PROGRESS " << int((fused ? 50 : 100) * (offset + count) / samples.size())
                      << '\n';
        }
        check(nemo_speech_diar_stream_finish(stream.get()));
        const double duration = samples.size() / 16000.0;
        auto segments = neuralSegments(stream.get(), duration);
        if (fused) {
            const auto acoustic =
                sona::diarizeRecording(samples, option("--models"), 0, 0.95f, false, 0.5, 0.5);
            segments = sona::fuseSpeakerEvidence(acoustic.segments, segments, duration);
        }
        auto result = snapshot(segments, model.get(), duration);
        if (args.contains("--frames")) {
            const auto first = nemo_speech_diar_frame_probs_start(stream.get());
            const auto count = nemo_speech_diar_frame_count(stream.get()) - first;
            std::vector<float> values(std::size_t(count) * nemo_speech_diar_num_speakers(model.get()));
            check(nemo_speech_diar_frame_probs(stream.get(), values.data(), values.size()));
            QJsonArray frames;
            for (float value : values)
                frames.append(value);
            result["frame_probabilities"] = frames;
            result["first_frame"] = double(first);
        }
        result["push_wall_seconds"] = timings;
        result["configuration"] =
            QJsonObject{{"preset", preset},
                        {"quantization", "Q8_0"},
                        {"device", gpu >= 0 ? "gpu" : "cpu"},
                        {"method", fused ? "neural_anchors_acoustic_completion" : "neural"},
                        {"onset", 0.5},
                        {"offset", 0.3},
                        {"padding_seconds", 0.03},
                        {"minimum_duration_seconds", 0.1},
                        {"minimum_gap_seconds", 0.1}};
        std::cout << QJsonDocument(result).toJson(QJsonDocument::Compact).constData() << '\n';
        return 0;
#endif
    } catch (const std::exception &error) {
        std::cout << QJsonDocument(QJsonObject{{"error", QString::fromUtf8(error.what())}})
                         .toJson(QJsonDocument::Compact)
                         .constData()
                  << '\n';
        return 1;
    }
}
