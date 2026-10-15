#pragma once
#include "analysis.hpp"

#include <QAudioFormat>
#include <QElapsedTimer>
#include <QObject>
#include <QString>
#include <QTimer>
#include <algorithm>
#include <deque>
#include <functional>
#include <memory>

class QAudioSource;
class QAudioSink;
class QIODevice;

namespace sona {
class PlaybackDevice;
class AudioEngine : public QObject {
  public:
    explicit AudioEngine(QObject *parent = nullptr);
    ~AudioEngine() override;
    void selectDemo(int profile);
    bool loadWave(const QString &path, QString &error);
    bool selectLoadedWave();
    bool seekFile(double seconds, QString &error);
    bool startMicrophone(QString &error);
    void stopMicrophone();
    void setPaused(bool paused);
    bool setAudible(bool audible, QString &error, bool rewind = true);
    void stopAudition();
    bool paused() const {
        return paused_;
    }
    bool live() const {
        return live_;
    }
    bool audible() const {
        return bool(sink_);
    }
    int profile() const {
        return profile_;
    }
    int sampleRate() const {
        return sampleRate_;
    }
    double elapsed() const {
        return elapsed_;
    }
    std::uint64_t sourceRevision() const {
        return sourceRevision_;
    }
    const Features &features() const {
        return features_;
    }
    QString sourceName() const;
    std::vector<float> recentAudio() const {
        const auto count = std::min(recent_.size(), std::size_t(sampleRate_) * 10);
        return {recent_.end() - std::ptrdiff_t(count), recent_.end()};
    }
    std::vector<float> conversationAudio() const {
        return {recent_.begin(), recent_.end()};
    }
    const std::vector<float> &loadedAudio() const {
        return fileSamples_;
    }
    double capturedSeconds() const {
        return double(capturedSamples_) / sampleRate_;
    }
    double filePosition() const {
        return fileSamples_.empty() ? 0 : double(cursor_ % fileSamples_.size()) / sampleRate_;
    }
    std::function<void(const Features &)> onFrame;
    std::function<void(const QString &)> onStatus;

  private:
    void tick();
    void ingest(const std::vector<float> &samples);
    void reset();
    QTimer timer_;
    QElapsedTimer clock_;
    qint64 lastTick_ = 0;
    double elapsed_ = 0;
    bool paused_ = false, live_ = false;
    int profile_ = 0, sampleRate_ = 24000, fileRate_ = 0;
    std::size_t cursor_ = 0;
    std::uint64_t capturedSamples_ = 0;
    std::uint64_t sourceRevision_ = 0;
    QString fileName_;
    std::vector<float> fileSamples_, history_, auditionDemo_;
    std::deque<float> recent_; // At most thirty seconds, in memory only.
    Analyzer analyzer_;
    Features features_;
    std::unique_ptr<QAudioSource> source_;
    std::unique_ptr<PlaybackDevice> playback_;
    std::unique_ptr<QAudioSink> sink_;
};
} // namespace sona
