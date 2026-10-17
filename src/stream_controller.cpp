#include "conversation_controller.hpp"
#include "stream_protocol.hpp"
#include <QJsonArray>
#include <QJsonDocument>
#include <QProcess>
#include <QTimer>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace sona {
void ConversationController::stream(const std::vector<float> &samples, int rate, double offset) {
    if (streamFailed_)
        return;
    if (rate < 8000 || rate > 96000 || samples.empty() || !std::isfinite(offset) || offset < 0) {
        setStatus("Unsupported live conversation audio");
        return;
    }
    if (process_ && (rate != streamRate_ || offset > submittedThrough_ + 1.5 / rate))
        reset();
    if (!process_) {
        streamOrigin_ = offset;
        submittedThrough_ = offset;
        streamRate_ = rate;
    }
    const auto skip =
        std::min(samples.size(), std::size_t(std::llround(std::max(0.0, submittedThrough_ - offset) * rate)));
    if (skip == samples.size())
        return;
    const std::vector<float> fresh(samples.begin() + std::ptrdiff_t(skip), samples.end());
    QByteArray packet;
    try {
        packet = encodeAudioChunk(fresh);
    } catch (const std::exception &error) {
        setStatus(QString::fromUtf8(error.what()));
        return;
    }
    submittedThrough_ = offset + double(samples.size()) / rate;
    waiting_ = true;
    if (process_) {
        auto *timeout = process_->findChild<QTimer *>();
        timeout->start(30000);
        process_->write(packet);
        return;
    }
    auto *process = new QProcess(this);
    process_ = process;
    auto *timeout = new QTimer(process);
    timeout->setSingleShot(true);
    connect(timeout, &QTimer::timeout, process, &QProcess::kill);
    timeout->start(30000);
    connect(process, &QProcess::started, this, [process, packet] { process->write(packet); });
    auto output = std::make_shared<QByteArray>();
    connect(process, &QProcess::readyReadStandardError, this, [process] { process->readAllStandardError(); });
    connect(process, &QProcess::readyReadStandardOutput, this, [this, process, timeout, output] {
        if (process_ != process)
            return;
        output->append(process->readAllStandardOutput());
        if (output->size() > 16 * 1024 * 1024) {
            process->kill();
            return;
        }
        qsizetype end = -1;
        while ((end = output->indexOf('\n')) >= 0) {
            const auto line = output->left(end);
            output->remove(0, end + 1);
            try {
                const auto result = parseDiarization(line);
                const auto object = QJsonDocument::fromJson(line).object();
                const auto position = object.value("offset_seconds");
                if (!position.isDouble() || !std::isfinite(position.toDouble()) || position.toDouble() < 0 ||
                    !result.sessionStable)
                    throw std::runtime_error("Invalid persistent conversation position");
                const double offset = streamOrigin_ + position.toDouble();
                const double through = offset + result.duration;
                if (through > submittedThrough_ + 0.02 || through + 0.02 < history_.analyzedThrough())
                    throw std::runtime_error("Persistent conversation returned invalid audio coverage");
                const auto scores = object.value("activity_scores").toArray();
                if (scores.size() > 8)
                    throw std::runtime_error("Invalid voice activity scores");
                std::vector<float> activity;
                for (const auto &score : scores) {
                    if (!score.isDouble() || !std::isfinite(score.toDouble()) || score.toDouble() < 0 ||
                        score.toDouble() > 1)
                        throw std::runtime_error("Invalid voice activity score");
                    activity.push_back(float(score.toDouble()));
                }
                history_.replace(result, offset, false);
                activityScores_ = std::move(activity);
                activityThrough_ = through;
                waiting_ = false;
                timeout->stop();
                setStatus("Live · persistent voice tracking");
            } catch (const std::exception &error) {
                waiting_ = false;
                streamFailed_ = true;
                process->kill();
                setStatus(QString::fromUtf8(error.what()));
                return;
            }
        }
    });
    connect(process, &QProcess::errorOccurred, this, [this, process](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart && process_ == process) {
            process_ = nullptr;
            waiting_ = false;
            streamFailed_ = true;
            process->deleteLater();
            setStatus("Persistent conversation worker could not start");
        }
    });
    connect(process, &QProcess::finished, this, [this, process](int, QProcess::ExitStatus) {
        if (process_ != process)
            return;
        process_ = nullptr;
        waiting_ = false;
        process->deleteLater();
        if (!streamFailed_)
            setStatus("Live tracking stopped · switch source to begin a new session");
        streamFailed_ = true;
    });
    process->setProgram(streamWorkerPath_);
    process->setArguments({"--stream", "--rate", QString::number(rate), "--models", modelDirectory_,
                           "--preset", "v3-streaming"});
    setStatus("Starting persistent voice tracking…");
    process->start();
}
} // namespace sona
