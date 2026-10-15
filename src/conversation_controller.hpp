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
                                    QString workerPath = {});
    ~ConversationController() override;
    void reset();
    void analyze(const std::vector<float> &samples, int rate, double offset, bool complete);
    bool busy() const {
        return process_ != nullptr;
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
    QString modelDirectory_, workerPath_, status_;
    QProcess *process_ = nullptr;
    ConversationHistory history_;
    int expectedSpeakers_ = 0;
};
} // namespace sona
