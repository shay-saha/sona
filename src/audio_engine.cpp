#include "audio_engine.hpp"
#include <QAudioDevice>
#include <QAudioSink>
#include <QAudioSource>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QMediaDevices>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace sona {
class PlaybackDevice : public QIODevice {
  public:
    PlaybackDevice(const std::vector<float> &samples, int rate, QAudioFormat format)
        : samples_(samples), rate_(rate), format_(format) {
        open(QIODevice::ReadOnly);
    }
    bool isSequential() const override {
        return true;
    }
    qint64 bytesAvailable() const override {
        return 65536 + QIODevice::bytesAvailable();
    }

  protected:
    qint64 readData(char *data, qint64 maxlen) override {
        const int frameBytes = format_.bytesPerFrame();
        const qint64 frames = maxlen / frameBytes;
        if (samples_.empty())
            return 0;
        for (qint64 i = 0; i < frames; ++i) {
            const auto a = std::size_t(position_) % samples_.size(), b = (a + 1) % samples_.size();
            const double fraction = position_ - std::floor(position_);
            const float value =
                float(std::clamp(samples_[a] * (1 - fraction) + samples_[b] * fraction, -1.0, 1.0));
            for (int c = 0; c < format_.channelCount(); ++c) {
                char *dest = data + i * frameBytes + c * format_.bytesPerSample();
                switch (format_.sampleFormat()) {
                case QAudioFormat::Float:
                    std::memcpy(dest, &value, 4);
                    break;
                case QAudioFormat::Int16: {
                    const auto s = std::int16_t(value * 32767);
                    std::memcpy(dest, &s, 2);
                    break;
                }
                case QAudioFormat::Int32: {
                    const auto s = std::int32_t(double(value) * 2147483647);
                    std::memcpy(dest, &s, 4);
                    break;
                }
                case QAudioFormat::UInt8:
                    *dest = char(std::clamp(int((value + 1) * 127.5), 0, 255));
                    break;
                default:
                    return 0;
                }
            }
            position_ += double(rate_) / format_.sampleRate();
            if (position_ >= samples_.size())
                position_ = std::fmod(position_, double(samples_.size()));
        }
        return frames * frameBytes;
    }
    qint64 writeData(const char *, qint64) override {
        return -1;
    }

  private:
    const std::vector<float> &samples_;
    int rate_;
    QAudioFormat format_;
    double position_ = 0;
};

AudioEngine::AudioEngine(QObject *parent) : QObject(parent) {
    history_.reserve(frameSize * 4);
    clock_.start();
    timer_.setInterval(33);
    timer_.setTimerType(Qt::PreciseTimer);
    connect(&timer_, &QTimer::timeout, this, [this] { tick(); });
    timer_.start();
}
AudioEngine::~AudioEngine() {
    stopAudition();
    if (source_)
        source_->stop();
}

void AudioEngine::reset() {
    ++sourceRevision_;
    cursor_ = 0;
    elapsed_ = 0;
    history_.clear();
    recent_.clear();
    analyzer_.reset();
    features_ = {};
    lastTick_ = clock_.elapsed();
    if (onFrame)
        onFrame(features_);
}
void AudioEngine::selectDemo(int profile) {
    stopMicrophone();
    stopAudition();
    profile_ = std::clamp(profile, 0, 2);
    sampleRate_ = 24000;
    paused_ = false;
    reset();
}
bool AudioEngine::loadWave(const QString &path, QString &error) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        error = file.errorString();
        return false;
    }
    if (file.size() > 100 * 1024 * 1024) {
        error = "Choose a WAV file smaller than 100 MB.";
        return false;
    }
    const QByteArray bytes = file.readAll();
    auto wave = decodeWave(std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
    if (!wave) {
        error = QString::fromStdString(wave.error);
        return false;
    }
    stopMicrophone();
    stopAudition();
    fileSamples_ = std::move(wave.samples);
    sampleRate_ = wave.sampleRate;
    fileRate_ = wave.sampleRate;
    fileName_ = QFileInfo(path).fileName();
    profile_ = -1;
    paused_ = false;
    reset();
    return true;
}
bool AudioEngine::selectLoadedWave() {
    if (fileSamples_.empty())
        return false;
    stopMicrophone();
    stopAudition();
    profile_ = -1;
    sampleRate_ = fileRate_;
    paused_ = false;
    reset();
    return true;
}
bool AudioEngine::startMicrophone(QString &error) {
    const auto device = QMediaDevices::defaultAudioInput();
    if (device.isNull()) {
        error = "No microphone is available. Connect one or use a demo / WAV file.";
        return false;
    }
    stopAudition();
    stopMicrophone();
    const auto format = device.preferredFormat();
    if (!format.isValid() || format.sampleRate() < 8000 || format.sampleRate() > 192000) {
        error = "This microphone's audio format is unsupported.";
        return false;
    }
    source_ = std::make_unique<QAudioSource>(device, format);
    sampleRate_ = format.sampleRate();
    live_ = true;
    paused_ = false;
    reset();
    auto *stream = source_->start();
    if (!stream || source_->error() != QAudio::NoError) {
        error = "Could not open the microphone. Check your system's microphone access settings.";
        stopMicrophone();
        selectDemo(0);
        return false;
    }
    auto pending = std::make_shared<QByteArray>();
    connect(stream, &QIODevice::readyRead, this, [this, stream, format, pending] {
        pending->append(stream->readAll());
        const auto frames = pending->size() / format.bytesPerFrame();
        std::vector<float> mono;
        mono.reserve(std::size_t(frames));
        for (qsizetype i = 0; i < frames; ++i) {
            double sum = 0;
            for (int c = 0; c < format.channelCount(); ++c)
                sum += format.normalizedSampleValue(pending->constData() + i * format.bytesPerFrame() +
                                                    c * format.bytesPerSample());
            mono.push_back(float(sum / format.channelCount()));
        }
        pending->remove(0, frames * format.bytesPerFrame());
        if (!paused_)
            ingest(mono);
    });
    connect(source_.get(), &QAudioSource::stateChanged, this, [this](QAudio::State state) {
        if (state == QAudio::StoppedState && live_ && source_ && source_->error() != QAudio::NoError) {
            // Defer destruction until the emitting audio object has returned.
            QTimer::singleShot(0, this, [this] {
                selectDemo(0);
                if (onStatus)
                    onStatus("Microphone disconnected. Switched to the warm synthetic demo.");
            });
        }
    });
    return true;
}
void AudioEngine::stopMicrophone() {
    live_ = false;
    if (source_) {
        source_->disconnect(this);
        source_->stop();
        source_.reset();
    }
}
void AudioEngine::setPaused(bool paused) {
    paused_ = paused;
    lastTick_ = clock_.elapsed();
    if (source_) {
        if (paused)
            source_->suspend();
        else
            source_->resume();
    }
    if (sink_) {
        if (paused)
            sink_->suspend();
        else
            sink_->resume();
    }
}
void AudioEngine::stopAudition() {
    if (sink_) {
        sink_->disconnect(this);
        sink_->stop();
        sink_.reset();
    }
    playback_.reset();
}
bool AudioEngine::setAudible(bool audible, QString &error) {
    stopAudition();
    if (!audible)
        return true;
    if (live_) {
        error = "Microphone monitoring is off to prevent feedback.";
        return false;
    }
    const auto device = QMediaDevices::defaultAudioOutput();
    if (device.isNull()) {
        error = "No audio output device is available. Visual analysis is still running.";
        return false;
    }
    const auto format = device.preferredFormat();
    if (!format.isValid()) {
        error = "No supported output format was found.";
        return false;
    }
    // Restart both timelines together so the displayed signal matches the audition.
    reset();
    if (profile_ >= 0)
        auditionDemo_ = synthesize(profile_, 0, std::size_t(sampleRate_) * 10, sampleRate_);
    const auto &samples = profile_ >= 0 ? auditionDemo_ : fileSamples_;
    playback_ = std::make_unique<PlaybackDevice>(samples, sampleRate_, format);
    sink_ = std::make_unique<QAudioSink>(device, format);
    sink_->setVolume(0.35);
    sink_->start(playback_.get());
    if (sink_->error() != QAudio::NoError) {
        error = "Audio output could not start.";
        stopAudition();
        return false;
    }
    connect(sink_.get(), &QAudioSink::stateChanged, this, [this](QAudio::State state) {
        if (state == QAudio::StoppedState && sink_ && sink_->error() != QAudio::NoError)
            QTimer::singleShot(0, this, [this] {
                stopAudition();
                if (onStatus)
                    onStatus("Audio output stopped. Check your sound device.");
            });
    });
    if (paused_)
        sink_->suspend();
    return true;
}
QString AudioEngine::sourceName() const {
    if (live_)
        return "Live microphone";
    if (profile_ < 0)
        return fileName_;
    const QStringList names{"Warm harmonics", "Bright harmonics", "Rhythmic pulses"};
    return names[profile_];
}
void AudioEngine::ingest(const std::vector<float> &samples) {
    recent_.insert(recent_.end(), samples.begin(), samples.end());
    const auto capacity = std::size_t(sampleRate_) * 10;
    if (recent_.size() > capacity)
        recent_.erase(recent_.begin(), recent_.begin() + std::ptrdiff_t(recent_.size() - capacity));
    history_.insert(history_.end(), samples.begin(), samples.end());
    if (history_.size() > frameSize * 2)
        history_.erase(history_.begin(), history_.end() - std::ptrdiff_t(frameSize * 2));
}
void AudioEngine::tick() {
    const auto now = clock_.elapsed();
    const double delta = std::clamp((now - lastTick_) / 1000.0, 0.0, 0.15);
    lastTick_ = now;
    if (paused_)
        return;
    elapsed_ += delta;
    if (!live_) {
        const auto count = std::max(std::size_t(1), std::size_t(delta * sampleRate_));
        if (profile_ >= 0)
            ingest(synthesize(profile_, cursor_, count, sampleRate_));
        else if (!fileSamples_.empty()) {
            std::vector<float> chunk(count);
            for (std::size_t i = 0; i < count; ++i)
                chunk[i] = fileSamples_[(cursor_ + i) % fileSamples_.size()];
            ingest(chunk);
        }
        cursor_ += count;
    }
    features_ = analyzer_.analyze(history_, sampleRate_);
    if (onFrame)
        onFrame(features_);
}
} // namespace sona
