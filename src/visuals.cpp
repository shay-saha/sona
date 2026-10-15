#include "visuals.hpp"
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QVector3D>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace sona {
namespace {
struct RenderVertex {
    float x, y, z, nx, ny, nz, stress;
};
QVector3D vec(Vec3 v) {
    return {float(v.x), float(v.y), float(v.z)};
}
constexpr auto vertexShader = R"(
#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in float stress;
uniform mat4 mvp;
out vec3 worldPosition;
out vec3 worldNormal;
out float deformation;
void main() {
    worldPosition=position; worldNormal=normal; deformation=stress;
    gl_Position=mvp*vec4(position,1.0);
})";
constexpr auto fragmentShader = R"(
#version 330 core
in vec3 worldPosition;
in vec3 worldNormal;
in float deformation;
uniform vec3 eye;
uniform int pass;
uniform int wireOnly;
out vec4 color;
void main() {
    if(pass==2) { color=vec4(0.19,0.21,0.23,0.42); return; }
    if(pass==1) { color=wireOnly==1 ? vec4(0.28,0.74,0.78,0.65) : vec4(0.02,0.12,0.15,0.30); return; }
    if(pass==3) { color=vec4(0.062,0.071,0.082,1.0); return; }
    vec3 n=normalize(worldNormal), v=normalize(eye-worldPosition);
    vec3 l=normalize(vec3(-3.0,5.0,4.0));
    float key=max(dot(n,l),0.0);
    float fill=max(dot(n,normalize(vec3(4.0,0.0,-3.0))),0.0);
    float rim=pow(1.0-max(dot(n,v),0.0),3.0);
    float spec=pow(max(dot(n,normalize(l+v)),0.0),48.0);
    float hot=clamp(deformation*1.6,0.0,1.0);
    vec3 base=mix(vec3(0.16,0.57,0.64),vec3(0.92,0.60,0.30),hot);
    vec3 shaded=base*(0.21+key*0.82+fill*0.18)+vec3(0.55,0.77,0.82)*spec*0.35+base*rim*0.24;
    color=vec4(shaded,1.0);
})";
QFont mono(int size = 11) {
    QFont f("DejaVu Sans Mono");
    f.setPixelSize(size);
    return f;
}
} // namespace
SurfaceView::SurfaceView(QWidget *parent) : QOpenGLWidget(parent) {
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAutoFillBackground(false);
    setMinimumSize(500, 340);
    setFocusPolicy(Qt::StrongFocus);
    setCursor(Qt::OpenHandCursor);
    setAccessibleName(
        "3D audio-driven spring surface. Drag to orbit, scroll to zoom, shift-click to apply an impulse.");
    clock_.start();
    timer_.setInterval(16);
    timer_.setTimerType(Qt::PreciseTimer);
    connect(&timer_, &QTimer::timeout, this, [this] { tick(); });
    timer_.start();
}
SurfaceView::~SurfaceView() {
    if (context()) {
        context()->disconnect(this);
        cleanup();
    }
}
void SurfaceView::cleanup() {
    makeCurrent();
    ready_ = false;
    meshVao_.destroy();
    gridVao_.destroy();
    vertices_.destroy();
    triangles_.destroy();
    edges_.destroy();
    grid_.destroy();
    program_.removeAllShaders();
    doneCurrent();
}
void SurfaceView::initializeGL() {
    if (!initializeOpenGLFunctions()) {
        if (onError)
            onError("OpenGL 3.3 is required.");
        return;
    }
    connect(
        context(), &QOpenGLContext::aboutToBeDestroyed, this, [this] { cleanup(); }, Qt::DirectConnection);
    rendererName_ = QString::fromLatin1(reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
    if (!program_.addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShader) ||
        !program_.addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShader) || !program_.link()) {
        if (onError)
            onError(program_.log());
        return;
    }
    meshVao_.create();
    meshVao_.bind();
    vertices_.create();
    vertices_.bind();
    vertices_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    vertices_.allocate(int(simulation_.mesh().directions.size() * sizeof(RenderVertex)));
    program_.bind();
    for (int i = 0; i < 3; ++i)
        program_.enableAttributeArray(i);
    program_.setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(RenderVertex));
    program_.setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3, sizeof(RenderVertex));
    program_.setAttributeBuffer(2, GL_FLOAT, 6 * sizeof(float), 1, sizeof(RenderVertex));
    triangles_.create();
    triangles_.bind();
    triangles_.allocate(simulation_.mesh().faces.data(),
                        int(simulation_.mesh().faces.size() * 3 * sizeof(std::uint32_t)));
    edges_.create();
    edges_.bind();
    edges_.allocate(simulation_.mesh().edges.data(),
                    int(simulation_.mesh().edges.size() * 2 * sizeof(std::uint32_t)));
    meshVao_.release();
    std::vector<RenderVertex> lines;
    for (int i = -10; i <= 10; ++i) {
        const float k = float(i);
        lines.insert(lines.end(), {{-10, -2.85f, k, 0, 1, 0, 0},
                                   {10, -2.85f, k, 0, 1, 0, 0},
                                   {k, -2.85f, -10, 0, 1, 0, 0},
                                   {k, -2.85f, 10, 0, 1, 0, 0}});
    }
    gridCount_ = int(lines.size());
    gridVao_.create();
    gridVao_.bind();
    grid_.create();
    grid_.bind();
    grid_.allocate(lines.data(), int(lines.size() * sizeof(RenderVertex)));
    program_.enableAttributeArray(0);
    program_.setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(RenderVertex));
    gridVao_.release();
    program_.release();
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);
    ready_ = true;
    dirty_ = true;
}
void SurfaceView::resizeGL(int, int) {}
void SurfaceView::uploadSurface() {
    const auto &mesh = simulation_.mesh();
    std::vector<QVector3D> positions(mesh.directions.size()), normals(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i)
        positions[i] = vec(mesh.directions[i]) * float(1.85 + simulation_.displacement()[i]);
    for (auto f : mesh.faces) {
        const auto n =
            QVector3D::crossProduct(positions[f[1]] - positions[f[0]], positions[f[2]] - positions[f[0]]);
        for (auto i : f)
            normals[i] += n;
    }
    std::vector<RenderVertex> data;
    data.reserve(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto p = positions[i], n = normals[i].normalized();
        data.push_back(
            {p.x(), p.y(), p.z(), n.x(), n.y(), n.z(),
             float(simulation_.displacement()[i] * 0.8 + std::abs(simulation_.velocity()[i]) * 0.035)});
    }
    vertices_.bind();
    vertices_.write(0, data.data(), int(data.size() * sizeof(RenderVertex)));
    dirty_ = false;
}
void SurfaceView::paintGL() {
    // Qt's compositor may change GL state between frames. Establish every state we depend on.
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClearDepth(1.0);
    glClearColor(0.062f, 0.071f, 0.082f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!ready_)
        return;
    // The viewport is opaque. Line antialiasing/blending must never lower its alpha.
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    const QVector3D eye(float(distance_ * std::sin(yaw_) * std::cos(pitch_)),
                        float(distance_ * std::sin(pitch_)),
                        float(distance_ * std::cos(yaw_) * std::cos(pitch_)));
    QMatrix4x4 projection, view;
    projection.perspective(38, float(width()) / std::max(1, height()), 0.1f, 100);
    view.lookAt(eye, {0, 0, 0}, {0, 1, 0});
    mvp_ = projection * view;
    program_.bind();
    program_.setUniformValue("mvp", mvp_);
    program_.setUniformValue("eye", eye);
    program_.setUniformValue("wireOnly", mode_ == 1 ? 1 : 0);
    if (gridVisible_) {
        program_.setUniformValue("pass", 2);
        gridVao_.bind();
        glDrawArrays(GL_LINES, 0, gridCount_);
        gridVao_.release();
    }
    meshVao_.bind();
    if (dirty_)
        uploadSurface();
    program_.setUniformValue("pass", mode_ == 1 ? 3 : 0);
    glDisable(GL_BLEND);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1, 1);
    triangles_.bind();
    glDrawElements(GL_TRIANGLES, GLsizei(simulation_.mesh().faces.size() * 3), GL_UNSIGNED_INT, nullptr);
    glDisable(GL_POLYGON_OFFSET_FILL);
    if (mode_ != 0) {
        glEnable(GL_BLEND);
        glDepthMask(GL_FALSE);
        program_.setUniformValue("pass", 1);
        edges_.bind();
        glDrawElements(GL_LINES, GLsizei(simulation_.mesh().edges.size() * 2), GL_UNSIGNED_INT, nullptr);
    }
    meshVao_.release();
    program_.release();
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDisable(GL_BLEND);
    ++frameCount_;
    const auto now = clock_.elapsed();
    if (now - fpsStart_ >= 500) {
        fps_ = frameCount_ * 1000.0 / double(now - fpsStart_);
        frameCount_ = 0;
        fpsStart_ = now;
    }
}
void SurfaceView::tick() {
    const auto now = clock_.elapsed();
    const double dt = (now - lastTick_) / 1000.0;
    lastTick_ = now;
    if (!isVisible())
        return;
    if (!paused_) {
        simulation_.advance(dt);
        dirty_ = true;
    }
    update();
}
void SurfaceView::setAudio(const Features &f) {
    simulation_.setDrive({f.active ? std::clamp((f.rmsDb + 55) / 43, 0.0, 1.0) : 0, f.angularity,
                          f.pitchHz > 0 ? 0.45 + f.pitchHz / 300 : 0.8});
}
void SurfaceView::setPaused(bool p) {
    paused_ = p;
    lastTick_ = clock_.elapsed();
}
void SurfaceView::singleStep() {
    if (paused_) {
        simulation_.singleStep();
        dirty_ = true;
        update();
    }
}
void SurfaceView::resetSimulation() {
    simulation_.reset();
    dirty_ = true;
    update();
}
void SurfaceView::resetCamera() {
    yaw_ = 0.5;
    pitch_ = 0.22;
    distance_ = 8.8;
    update();
}
void SurfaceView::setRenderMode(int mode) {
    mode_ = std::clamp(mode, 0, 2);
    update();
}
void SurfaceView::mousePressEvent(QMouseEvent *e) {
    if (e->button() != Qt::LeftButton)
        return;
    drag_ = e->position();
    setFocus();
    setCursor(Qt::ClosedHandCursor);
    if (e->modifiers() & Qt::ShiftModifier) {
        double best = 1e9;
        std::size_t index = 0;
        for (std::size_t i = 0; i < simulation_.mesh().directions.size(); ++i) {
            auto d = simulation_.mesh().directions[i];
            if (d.x * std::sin(yaw_) * std::cos(pitch_) + d.y * std::sin(pitch_) +
                    d.z * std::cos(yaw_) * std::cos(pitch_) <=
                0)
                continue;
            const auto clip = mvp_ * QVector4D(vec(d) * float(1.85 + simulation_.displacement()[i]), 1);
            if (clip.w() <= 0)
                continue;
            const QPointF pixel((clip.x() / clip.w() + 1) * width() / 2,
                                (1 - clip.y() / clip.w()) * height() / 2);
            const auto delta = pixel - e->position();
            const double distance = delta.x() * delta.x() + delta.y() * delta.y();
            if (distance < best) {
                best = distance;
                index = i;
            }
        }
        if (best < 2500) {
            simulation_.impulse(index, 5);
            dirty_ = true;
            update();
        }
    }
}
void SurfaceView::mouseMoveEvent(QMouseEvent *e) {
    if (e->buttons() & Qt::LeftButton) {
        auto d = e->position() - drag_;
        yaw_ -= d.x() * 0.006;
        pitch_ = std::clamp(pitch_ + d.y() * 0.006, -1.4, 1.4);
        drag_ = e->position();
        update();
    }
}
void SurfaceView::mouseReleaseEvent(QMouseEvent *) {
    setCursor(Qt::OpenHandCursor);
}
void SurfaceView::mouseDoubleClickEvent(QMouseEvent *) {
    resetCamera();
}
void SurfaceView::wheelEvent(QWheelEvent *e) {
    distance_ = std::clamp(distance_ * std::exp(-e->angleDelta().y() * 0.001), 5.0, 22.0);
    update();
    e->accept();
}

SpectrumCanvas::SpectrumCanvas(QWidget *parent)
    : QWidget(parent), history_(480, int(spectrumBands), QImage::Format_RGB32) {
    setMinimumHeight(110);
    setAccessibleName("Logarithmic spectrogram and waveform");
    clear();
}
void SpectrumCanvas::clear() {
    history_.fill(QColor("#11151a"));
    frame_ = {};
    update();
}
void SpectrumCanvas::setFrame(const Features &f, int rate) {
    frame_ = f;
    rate_ = rate;
    for (int y = 0; y < history_.height(); ++y) {
        auto *row = reinterpret_cast<QRgb *>(history_.scanLine(y));
        std::memmove(row, row + 1, std::size_t(history_.width() - 1) * sizeof(QRgb));
        const double v = f.spectrum[std::size_t(history_.height() - 1 - y)];
        row[history_.width() - 1] =
            QColor::fromRgbF(0.06 + 0.80 * v * v, 0.08 + 0.62 * v, 0.11 + 0.42 * v).rgb();
    }
    update();
}
void SpectrumCanvas::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.fillRect(rect(), QColor("#11151a"));
    p.setFont(mono(10));
    p.setPen(QColor("#9ba6b0"));
    const double w = width(), h = height();
    const QRectF area(45, 8, w * 0.7 - 55, h - 29);
    p.drawImage(area, history_);
    p.drawText(QRectF(3, 6, 35, 16), Qt::AlignRight,
               QString::number(std::min(12000, rate_ / 2) / 1000) + "k");
    p.drawText(QRectF(3, h - 36, 35, 16), Qt::AlignRight, "60");
    p.drawText(QRectF(45, h - 19, 300, 16), "Hz (log)       amplitude: -80 to 0 dBFS");
    const QRectF wave(w * 0.73, 8, w * 0.25, h - 29);
    p.setPen(QColor("#35414a"));
    p.drawLine(QPointF(wave.left(), wave.center().y()), QPointF(wave.right(), wave.center().y()));
    QPainterPath path;
    for (std::size_t i = 0; i < frame_.waveform.size(); ++i) {
        QPointF point(wave.left() + wave.width() * i / (frame_.waveform.size() - 1),
                      wave.center().y() - frame_.waveform[i] * wave.height());
        if (i == 0)
            path.moveTo(point);
        else
            path.lineTo(point);
    }
    p.setRenderHint(QPainter::Antialiasing);
    p.setPen(QColor("#79bec8"));
    p.drawPath(path);
    p.setPen(QColor("#9ba6b0"));
    p.drawText(QRectF(wave.left(), h - 19, wave.width(), 16),
               "Waveform / " + QString::number(1000.0 * sona::frameSize / rate_, 'f', 0) + " ms");
}
ShapeCanvas::ShapeCanvas(bool angular, QWidget *parent) : QWidget(parent), angular_(angular) {
    setMinimumSize(180, 190);
    setAccessibleName(angular ? "Angular shape" : "Rounded shape");
}
void ShapeCanvas::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const QPointF c(width() / 2.0, height() / 2.0);
    const double radius = std::min(width(), height()) * 0.34;
    QPainterPath path;
    const int n = angular_ ? 24 : 240;
    for (int i = 0; i <= n; ++i) {
        const double a = 2 * pi * i / n - pi / 2;
        const double r = radius * (angular_ ? (i % 2 ? 0.5 : 1.1) : (1 + 0.15 * std::cos(5 * a)));
        const auto q = c + QPointF(std::cos(a) * r, std::sin(a) * r);
        if (i == 0)
            path.moveTo(q);
        else
            path.lineTo(q);
    }
    path.closeSubpath();
    p.setBrush(QColor("#719ca5"));
    p.setPen(QPen(QColor("#b0cbd0"), 1));
    p.drawPath(path);
}
} // namespace sona
