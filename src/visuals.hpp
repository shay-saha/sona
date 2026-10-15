#pragma once
#include "analysis.hpp"
#include "simulation.hpp"
#include <QElapsedTimer>
#include <QImage>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QTimer>
#include <functional>

namespace sona {
class SurfaceView : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
  public:
    explicit SurfaceView(QWidget *parent = nullptr);
    ~SurfaceView() override;
    void setAudio(const Features &features);
    void setPaused(bool paused);
    void singleStep();
    void resetSimulation();
    void resetCamera();
    void setRenderMode(int mode);
    void setGridVisible(bool visible) {
        gridVisible_ = visible;
        update();
    }
    void setParameters(SimulationParameters p) {
        simulation_.setParameters(p);
    }
    const SurfaceSimulation &simulation() const {
        return simulation_;
    }
    bool paused() const {
        return paused_;
    }
    bool rendererReady() const {
        return ready_;
    }
    int renderMode() const {
        return mode_;
    }
    QString rendererName() const {
        return rendererName_;
    }
    double fps() const {
        return fps_;
    }
    std::function<void(const QString &)> onError;

  protected:
    void initializeGL() override;
    void resizeGL(int, int) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent *) override;
    void mouseMoveEvent(QMouseEvent *) override;
    void mouseReleaseEvent(QMouseEvent *) override;
    void mouseDoubleClickEvent(QMouseEvent *) override;
    void wheelEvent(QWheelEvent *) override;

  private:
    void cleanup();
    void uploadSurface();
    void tick();
    SurfaceSimulation simulation_;
    QTimer timer_;
    QElapsedTimer clock_;
    qint64 lastTick_ = 0, fpsStart_ = 0;
    int frameCount_ = 0, mode_ = 2;
    double fps_ = 0, yaw_ = 0.5, pitch_ = 0.22, distance_ = 8.8;
    bool paused_ = false, ready_ = false, dirty_ = true, gridVisible_ = true;
    QPointF drag_;
    QMatrix4x4 mvp_;
    QString rendererName_;
    QOpenGLShaderProgram program_;
    QOpenGLVertexArrayObject meshVao_, gridVao_;
    QOpenGLBuffer vertices_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer triangles_{QOpenGLBuffer::IndexBuffer};
    QOpenGLBuffer edges_{QOpenGLBuffer::IndexBuffer};
    QOpenGLBuffer grid_{QOpenGLBuffer::VertexBuffer};
    int gridCount_ = 0;
};
class SpectrumCanvas : public QWidget {
  public:
    explicit SpectrumCanvas(QWidget *parent = nullptr);
    void setFrame(const Features &features, int rate);
    void clear();

  protected:
    void paintEvent(QPaintEvent *) override;

  private:
    Features frame_;
    int rate_ = 24000;
    QImage history_;
};
class ShapeCanvas : public QWidget {
  public:
    explicit ShapeCanvas(bool angular, QWidget *parent = nullptr);

  protected:
    void paintEvent(QPaintEvent *) override;

  private:
    bool angular_;
};
} // namespace sona
