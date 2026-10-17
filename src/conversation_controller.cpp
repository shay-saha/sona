#include "conversation_controller.hpp"
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QProcess>
#include <QTimer>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace sona {
ConversationController::ConversationController(QString directory, QObject *parent, QString workerPath,
                                               QString streamWorkerPath)
    : QObject(parent), modelDirectory_(std::move(directory)), workerPath_(std::move(workerPath)),
      streamWorkerPath_(std::move(streamWorkerPath)) {
    const bool customWorker = !workerPath_.isEmpty();
    if (workerPath_.isEmpty())
        workerPath_ = QDir(QCoreApplication::applicationDirPath()).filePath("sona_diarize");
    const bool customStreamWorker = !streamWorkerPath_.isEmpty();
    if (streamWorkerPath_.isEmpty())
        streamWorkerPath_ = QDir(QCoreApplication::applicationDirPath()).filePath("sona_stream");
    neuralAvailable_ =
        (!customWorker || customStreamWorker) &&
        QFileInfo::exists(QDir(modelDirectory_).filePath("Nemotron-3-Diarization.q8_0.gguf")) &&
        QProcess::execute(streamWorkerPath_, {"--available"}) == 0;
}
ConversationController::~ConversationController() {
    onUpdate = {};
    reset();
}
void ConversationController::setStatus(const QString &message) {
    status_ = message;
    if (onUpdate)
        onUpdate();
}
void ConversationController::reset() {
    if (process_) {
        process_->disconnect(this);
        process_->kill();
        process_->waitForFinished(250);
        process_->deleteLater();
        process_ = nullptr;
    }
    history_.clear();
    waiting_ = false;
    streamFailed_ = false;
    submittedThrough_ = streamOrigin_ = activityThrough_ = 0;
    streamRate_ = 0;
    activityScores_.clear();
    setStatus("Open a conversation WAV or enable the microphone");
}
void ConversationController::analyze(const std::vector<float> &samples, int rate, double offset,
                                     bool complete) {
    if (busy())
        return;
    if (usesStreaming() && !complete) {
        stream(samples, rate, offset);
        return;
    }
    if (rate < 8000 || rate > 192000 || samples.size() < std::size_t(rate) ||
        samples.size() > 50 * 1024 * 1024 || !std::isfinite(offset) || offset < 0) {
        setStatus("Insufficient or unsupported conversation audio");
        return;
    }
    if (!QFileInfo::exists(QDir(modelDirectory_).filePath("nemo_en_titanet_small.onnx"))) {
        setStatus("Conversation models unavailable · run research/scripts/fetch_conversation.py");
        return;
    }
    auto *process = new QProcess(this);
    process_ = process;
    waiting_ = true;
    auto *timeout = new QTimer(process);
    timeout->setSingleShot(true);
    connect(timeout, &QTimer::timeout, process, &QProcess::kill);
    timeout->start(10 * 60 * 1000);
    auto output = std::make_shared<QByteArray>();
    auto payload = std::make_shared<QByteArray>(reinterpret_cast<const char *>(samples.data()),
                                                qsizetype(samples.size() * sizeof(float)));
    connect(process, &QProcess::started, this, [process, payload] {
        process->write(*payload);
        payload->clear();
        payload->squeeze();
        process->closeWriteChannel();
    });
    connect(process, &QProcess::readyReadStandardOutput, this, [process, output] {
        output->append(process->readAllStandardOutput());
        if (output->size() > 16 * 1024 * 1024)
            process->kill();
    });
    auto progress = std::make_shared<QByteArray>();
    connect(process, &QProcess::readyReadStandardError, this, [this, process, progress, complete, timeout] {
        progress->append(process->readAllStandardError());
        int end = -1;
        while ((end = int(progress->indexOf('\n'))) >= 0) {
            const auto line = progress->left(end);
            progress->remove(0, end + 1);
            if (line.startsWith("SONA_PROGRESS ")) {
                bool valid = false;
                const int percent = line.mid(14).toInt(&valid);
                if (valid && percent >= 0 && percent <= 100) {
                    timeout->start(30000);
                    setStatus(QString(complete ? "Analyzing recording · %1%" : "Analyzing conversation · %1%")
                                  .arg(percent));
                }
            }
        }
        if (progress->size() > 4096)
            progress->clear();
    });
    connect(process, &QProcess::errorOccurred, this, [this, process](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart && process_ == process) {
            process_ = nullptr;
            waiting_ = false;
            process->deleteLater();
            setStatus("Conversation worker could not start · build the sona_diarize target");
        }
    });
    connect(process, &QProcess::finished, this,
            [this, process, output, offset, complete,
             duration = double(samples.size()) / rate](int exitCode, QProcess::ExitStatus exitStatus) {
                if (process_ != process)
                    return;
                output->append(process->readAllStandardOutput());
                process_ = nullptr;
                waiting_ = false;
                process->deleteLater();
                try {
                    if (exitStatus != QProcess::NormalExit || output->isEmpty())
                        throw std::runtime_error("Conversation worker stopped before producing a result");
                    const auto result = parseDiarization(*output);
                    if (std::abs(result.duration - duration) > 0.02)
                        throw std::runtime_error("Conversation worker returned incomplete audio coverage");
                    if (exitCode != 0)
                        throw std::runtime_error("Conversation analysis failed");
                    history_.replace(result, offset, complete);
                    setStatus(complete ? "Recording analyzed · anonymous speaker assignments"
                                       : "Live · provisional assignments · 30 s analysis window");
                } catch (const std::exception &e) {
                    setStatus(QString::fromUtf8(e.what()));
                }
            });
    process->setProgram(usesStreaming() ? streamWorkerPath_ : workerPath_);
    QStringList arguments{"--stdin", "--rate", QString::number(rate), "--models", modelDirectory_};
    if (usesStreaming())
        arguments << "--preset" << "v3-offline";
    else
        arguments << "--speakers" << QString::number(expectedSpeakers_);
    process->setArguments(arguments);
    setStatus(complete ? "Analyzing recording…" : "Analyzing recent conversation…");
    process->start();
}
} // namespace sona
