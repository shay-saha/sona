#pragma once
#include "analysis.hpp"
#include "conversation.hpp"
#include "human_mesh.hpp"
#include <QElapsedTimer>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QTimer>
#include <functional>

namespace sona {
class PeopleView : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
  public:
    explicit PeopleView(QWidget *parent = nullptr);
    ~PeopleView() override;
    void setAudio(const Features &features, double sourceTime, bool synthetic);
    void setHeightEstimate(double centimetres, double pilotMae, double lower = 0, double upper = 0);
    void clearHeightEstimate();
    void setShoulderEstimate(double centimetres);
    void clearShoulderEstimate();
    double shoulderEstimateCm() const {
        return shoulderCm_;
    }
    void setProfileSummary(const QString &text) {
        profileSummary_ = text;
        update();
    }
    void resetFormation();
    void resetCamera();
    void setPaused(bool paused);
    void singleStep();
    void setRenderMode(int mode);
    void setGridVisible(bool visible);
    void enableConversation();
    void setConversationData(const std::vector<SpeechSegment> &segments, double analyzedThrough,
                             const std::vector<TimeSpan> &coverage = {});
    void setConversationTime(double seconds, bool seekable);
    void setConversationStatus(const QString &status);
    void setConversationActivity(const std::vector<float> &scores, double through);
    const ConversationSummary &conversationSummary() const {
        return conversationSummary_;
    }
    std::vector<QVector3D> conversationPositions() const;
    std::function<void(double)> onSeek;
    const HumanMesh &mesh() const {
        return mesh_;
    }
    bool rendererReady() const {
        return ready_;
    }
    bool paused() const {
        return paused_;
    }
    bool hasHeightEstimate() const {
        return heightCm_ > 0;
    }
    double heightEstimateCm() const {
        return heightCm_;
    }
    double pilotMaeCm() const {
        return pilotMae_;
    }
    double formation() const {
        return formation_;
    }
    double observedSeconds() const {
        return audioSeconds_;
    }
    double heightScale() const {
        return scale_;
    }
    double fps() const {
        return fps_;
    }
    std::function<void(const QString &)> onError;

  protected:
    void initializeGL() override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent *) override;
    void mouseMoveEvent(QMouseEvent *) override;
    void mouseReleaseEvent(QMouseEvent *) override;
    void mouseDoubleClickEvent(QMouseEvent *) override;
    void wheelEvent(QWheelEvent *) override;

  private:
    void cleanup();
    void tick();
    void advance(double seconds);
    void drawLabels();
    void updateProportions();
    void drawConversationLabels(const QMatrix4x4 &mvp, int sceneHeight);
    void drawConnections(const QMatrix4x4 &mvp);
    int timelineHeight() const;
    bool conversation_ = false, seekable_ = false;
    double conversationTime_ = 0, analyzedThrough_ = 0;
    QString conversationStatus_;
    std::vector<SpeechSegment> conversationSegments_;
    std::vector<TimeSpan> conversationCoverage_;
    ConversationSummary conversationSummary_;
    std::vector<float> activityScores_;
    double activityThrough_ = 0;
    HumanMesh mesh_, referenceMesh_;
    QString profileSummary_;
    Features audio_;
    QTimer timer_;
    QElapsedTimer clock_;
    qint64 lastTick_ = 0, fpsStart_ = 0;
    int frames_ = 0, mode_ = 2, gridCount_ = 0;
    double time_ = 0, audioTime_ = 0, audioSeconds_ = 0, formation_ = 0;
    double heightCm_ = 0, pilotMae_ = 0, scale_ = 1, activity_ = 0, fps_ = 0;
    double heightLower_ = 0, heightUpper_ = 0;
    double shoulderCm_ = 0, shoulderFactor_ = 1;
    double yaw_ = 0.3, pitch_ = 0.08, distance_ = 3.4;
    bool ready_ = false, paused_ = false, gridVisible_ = true, synthetic_ = true;
    QPointF drag_, pan_;
    QOpenGLShaderProgram program_;
    QOpenGLVertexArrayObject bodyVao_, gridVao_, linksVao_;
    QOpenGLBuffer links_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer vertices_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer triangles_{QOpenGLBuffer::IndexBuffer};
    QOpenGLBuffer edges_{QOpenGLBuffer::IndexBuffer};
    QOpenGLBuffer grid_{QOpenGLBuffer::VertexBuffer};
};
} // namespace sona
