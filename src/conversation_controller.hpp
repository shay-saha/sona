#pragma once
#include "conversation.hpp"
#include <QObject>
#include <QString>
#include <functional>

class QProcess;
namespace sona {
class ConversationController : public QObject {
  public:
    explicit ConversationController(QString modelDirectory, QObject *parent = nullptr,
                                    QString workerPath = {}, QString streamWorkerPath = {});
    ~ConversationController() override;
    void reset();
    void analyze(const std::vector<float> &samples, int rate, double offset, bool complete);
    bool busy() const {
        return waiting_;
    }
    bool usesStreaming() const {
        return neuralAvailable_ && expectedSpeakers_ == 0;
    }
    const std::vector<float> &activityScores() const {
        return activityScores_;
    }
    double activityThrough() const {
        return activityThrough_;
    }
    const ConversationHistory &history() const {
        return history_;
    }
    const QString &status() const {
        return status_;
    }
    void setExpectedSpeakers(int count) {
        expectedSpeakers_ = count;
    }
    std::function<void()> onUpdate;

  private:
    void setStatus(const QString &message);
    void stream(const std::vector<float> &samples, int rate, double offset);
    QString modelDirectory_, workerPath_, streamWorkerPath_, status_;
    QProcess *process_ = nullptr;
    ConversationHistory history_;
    int expectedSpeakers_ = 0;
    bool neuralAvailable_ = false, waiting_ = false, streamFailed_ = false;
    int streamRate_ = 0;
    double submittedThrough_ = 0, streamOrigin_ = 0, activityThrough_ = 0;
    std::vector<float> activityScores_;
};
} // namespace sona
