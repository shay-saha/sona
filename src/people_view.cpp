#include "people_view.hpp"
#include <QMouseEvent>
#include <QPainter>
#include <QVector3D>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

namespace sona {
namespace {
struct Vertex {
    float x, y, z, nx, ny, nz;
};
constexpr auto vertexShader = R"(
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
uniform mat4 mvp;
uniform float stature;
uniform float formation;
uniform float time;
uniform float activity;
uniform float pointScale;
uniform int pass;
out vec3 worldPosition;
out vec3 worldNormal;
out float bodyHeight;
out float assembly;
float random(float seed) { return fract(sin(seed * 12.9898) * 43758.5453); }
void main() {
    bodyHeight = clamp(position.y / 1.75, 0.0, 1.0);
    assembly = smoothstep(bodyHeight * 0.8, bodyHeight * 0.8 + 0.2, formation);
    vec3 p = position;
    if (pass == 2) {
        float seed = float(gl_VertexID) + 1.0;
        vec3 scatter = vec3(random(seed), random(seed + 17.0), random(seed + 73.0)) - 0.5;
        float orbit = time * 0.18;
        scatter.xz = mat2(cos(orbit), -sin(orbit), sin(orbit), cos(orbit)) * scatter.xz;
        p += scatter * (1.0 - assembly) * 0.9;
    } else if (pass != 3) {
        p += normal * activity * 0.0015 * sin(position.y * 28.0 - time * 4.0);
    }
    worldPosition = pass == 3 ? p : p * stature;
    worldNormal = normal;
    gl_Position = mvp * vec4(worldPosition, 1.0);
    gl_PointSize = pointScale * (1.3 + 0.8 * (1.0 - assembly));
})";
constexpr auto fragmentShader = R"(
#version 330 core
in vec3 worldPosition;
in vec3 worldNormal;
in float bodyHeight;
in float assembly;
uniform vec3 eye;
uniform float formation;
uniform float time;
uniform float activity;
uniform float brightness;
uniform int pass;
out vec4 color;
void main() {
    if (pass == 3) {
        float fade = 1.0 - smoothstep(0.3, 1.9, length(worldPosition.xz));
        color = vec4(0.22, 0.27, 0.3, 0.32 * fade);
        return;
    }
    if (pass == 2) {
        float radius = length(gl_PointCoord - 0.5) * 2.0;
        if (radius > 1.0 || assembly > 0.99) discard;
        float intensity = (1.0 - radius) * (0.3 + 0.6 * formation);
        color = vec4(mix(vec3(0.2, 0.44, 0.49), vec3(0.5, 0.9, 0.88), assembly), intensity);
        return;
    }
    float frontier = formation * 1.08 - 0.04;
    if (bodyHeight > frontier) discard;
    if (pass == 4) { color = vec4(0.062, 0.071, 0.082, 1.0); return; }
    float scan = exp(-pow((bodyHeight - frontier + 0.012) * 70.0, 2.0));
    if (pass == 1) {
        color = vec4(0.13, 0.34, 0.37, 0.24);
        return;
    }
    vec3 n = normalize(worldNormal);
    vec3 view = normalize(eye - worldPosition);
    vec3 light = normalize(vec3(-2.0, 3.5, 4.0));
    float key = max(dot(n, light), 0.0);
    float fill = max(dot(n, normalize(vec3(3.0, 1.5, -2.0))), 0.0);
    float rim = pow(1.0 - max(dot(n, view), 0.0), 3.0);
    float specular = pow(max(dot(n, normalize(light + view)), 0.0), 52.0);
    float wave = pow(0.5 + 0.5 * sin(worldPosition.y * 30.0 - time * 4.0), 12.0) * activity;
    vec3 base = mix(vec3(0.52, 0.63, 0.65), vec3(0.34, 0.61, 0.64), brightness * 0.4);
    vec3 shaded = base * (0.2 + 0.72 * key + 0.23 * fill);
    shaded += vec3(0.32, 0.66, 0.7) * rim * 0.4 + vec3(0.7, 0.87, 0.88) * specular * 0.3;
    shaded += vec3(0.16, 0.67, 0.63) * (scan * 0.7 + wave * 0.1);
    color = vec4(shaded, 1.0);
})";
} // namespace

PeopleView::PeopleView(QWidget *parent) : QOpenGLWidget(parent), mesh_(makeHumanMesh()) {
    referenceMesh_ = mesh_;
    setObjectName("peopleView");
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAutoFillBackground(false);
    setMinimumSize(500, 340);
    setFocusPolicy(Qt::StrongFocus);
    setCursor(Qt::OpenHandCursor);
    setAccessibleName("3D human audio portrait. Drag to orbit, right-drag to pan, wheel to zoom.");
    clock_.start();
    timer_.setInterval(16);
    timer_.setTimerType(Qt::PreciseTimer);
    connect(&timer_, &QTimer::timeout, this, [this] { tick(); });
    timer_.start();
}
PeopleView::~PeopleView() {
    if (context()) {
        context()->disconnect(this);
        cleanup();
    }
}
void PeopleView::cleanup() {
    makeCurrent();
    ready_ = false;
    bodyVao_.destroy();
    gridVao_.destroy();
    vertices_.destroy();
    triangles_.destroy();
    edges_.destroy();
    grid_.destroy();
    program_.removeAllShaders();
    doneCurrent();
}
void PeopleView::initializeGL() {
    if (!initializeOpenGLFunctions()) {
        if (onError)
            onError("OpenGL 3.3 is required for the People view.");
        return;
    }
    connect(
        context(), &QOpenGLContext::aboutToBeDestroyed, this, [this] { cleanup(); }, Qt::DirectConnection);
    if (!program_.addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShader) ||
        !program_.addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShader) || !program_.link()) {
        if (onError)
            onError(program_.log());
        return;
    }
    std::vector<Vertex> data;
    data.reserve(mesh_.positions.size());
    for (std::size_t i = 0; i < mesh_.positions.size(); ++i) {
        const auto p = mesh_.positions[i], n = mesh_.normals[i];
        data.push_back({float(p.x), float(p.y), float(p.z), float(n.x), float(n.y), float(n.z)});
    }
    program_.bind();
    bodyVao_.create();
    bodyVao_.bind();
    vertices_.create();
    vertices_.bind();
    vertices_.allocate(data.data(), int(data.size() * sizeof(Vertex)));
    program_.enableAttributeArray(0);
    program_.enableAttributeArray(1);
    program_.setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(Vertex));
    program_.setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3, sizeof(Vertex));
    triangles_.create();
    triangles_.bind();
    triangles_.allocate(mesh_.faces.data(), int(mesh_.faces.size() * 3 * sizeof(std::uint32_t)));
    edges_.create();
    edges_.bind();
    edges_.allocate(mesh_.edges.data(), int(mesh_.edges.size() * 2 * sizeof(std::uint32_t)));
    bodyVao_.release();
    data.clear();
    for (int line = -10; line <= 10; ++line) {
        const float at = line * 0.15f;
        data.insert(data.end(), {{-1.5f, -0.004f, at, 0, 1, 0},
                                 {1.5f, -0.004f, at, 0, 1, 0},
                                 {at, -0.004f, -1.5f, 0, 1, 0},
                                 {at, -0.004f, 1.5f, 0, 1, 0}});
    }
    gridCount_ = int(data.size());
    gridVao_.create();
    gridVao_.bind();
    grid_.create();
    grid_.bind();
    grid_.allocate(data.data(), int(data.size() * sizeof(Vertex)));
    program_.enableAttributeArray(0);
    program_.enableAttributeArray(1);
    program_.setAttributeBuffer(0, GL_FLOAT, 0, 3, sizeof(Vertex));
    program_.setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3, sizeof(Vertex));
    gridVao_.release();
    program_.release();
    glEnable(GL_PROGRAM_POINT_SIZE);
    ready_ = true;
}
void PeopleView::paintGL() {
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glClearDepth(1);
    glClearColor(0.062f, 0.071f, 0.082f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!ready_)
        return;
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    const QVector3D target(float(pan_.x()), float(0.88 + pan_.y()), 0);
    const auto eye = target + QVector3D(float(distance_ * std::sin(yaw_) * std::cos(pitch_)),
                                        float(distance_ * std::sin(pitch_)),
                                        float(distance_ * std::cos(yaw_) * std::cos(pitch_)));
    QMatrix4x4 projection, camera;
    projection.perspective(38, float(width()) / std::max(1, height()), 0.02f, 30);
    camera.lookAt(eye, target, {0, 1, 0});
    const auto mvp = projection * camera;
    program_.bind();
    program_.setUniformValue("mvp", mvp);
    program_.setUniformValue("eye", eye);
    program_.setUniformValue("stature", float(scale_));
    program_.setUniformValue("formation", float(formation_));
    program_.setUniformValue("time", float(time_));
    program_.setUniformValue("activity", float(activity_));
    program_.setUniformValue("brightness", float(std::clamp((audio_.centroidHz - 300) / 4000, 0.0, 1.0)));
    program_.setUniformValue("pointScale", float(devicePixelRatioF()));
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    if (gridVisible_) {
        program_.setUniformValue("pass", 3);
        gridVao_.bind();
        glDrawArrays(GL_LINES, 0, gridCount_);
        gridVao_.release();
    }
    bodyVao_.bind();
    glDisable(GL_BLEND);
    program_.setUniformValue("pass", mode_ == 1 ? 4 : 0);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1, 1);
    triangles_.bind();
    glDrawElements(GL_TRIANGLES, GLsizei(mesh_.faces.size() * 3), GL_UNSIGNED_INT, nullptr);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);
    glDisable(GL_POLYGON_OFFSET_FILL);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    if (mode_ != 0) {
        program_.setUniformValue("pass", 1);
        edges_.bind();
        glDrawElements(GL_LINES, GLsizei(mesh_.edges.size() * 2), GL_UNSIGNED_INT, nullptr);
    }
    program_.setUniformValue("pass", 2);
    glDrawArrays(GL_POINTS, 0, GLsizei(mesh_.positions.size()));
    bodyVao_.release();
    program_.release();
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    drawLabels();
    ++frames_;
    const auto now = clock_.elapsed();
    if (now - fpsStart_ >= 500) {
        fps_ = frames_ * 1000.0 / double(now - fpsStart_);
        frames_ = 0;
        fpsStart_ = now;
    }
}
void PeopleView::drawLabels() {
    QPainter painter(this);
    auto font = painter.font();
    font.setPixelSize(12);
    painter.setFont(font);
    painter.setPen(QColor("#b6c4ca"));
    painter.drawText(20, 29,
                     synthetic_ ? "Synthetic source · Generic figure" : "Generic anatomy · Single speaker");
    painter.setPen(QColor("#788b94"));
    painter.drawText(20, 49, QString("Formation · %1 s active audio").arg(audioSeconds_, 0, 'f', 1));
    if (!profileSummary_.isEmpty())
        painter.drawText(20, height() - 47, profileSummary_);
    if (hasHeightEstimate()) {
        painter.setPen(QColor("#9ed5d0"));
        const auto uncertainty = heightUpper_ > heightLower_
                                     ? QString("Nominal 90% range %1–%2 cm")
                                           .arg(heightLower_, 0, 'f', 1)
                                           .arg(heightUpper_, 0, 'f', 1)
                                     : QString("Pilot MAE %1 cm").arg(pilotMae_, 0, 'f', 1);
        painter.drawText(
            20, height() - 25,
            QString("Experimental height %1 cm · %2").arg(heightCm_, 0, 'f', 1).arg(uncertainty));
    } else {
        painter.drawText(20, height() - 25,
                         synthetic_ ? "Generic proportions" : "Height unavailable · Generic proportions");
    }
}
void PeopleView::setAudio(const Features &features, double sourceTime, bool synthetic) {
    const double elapsed = std::clamp(sourceTime - audioTime_, 0.0, 0.15);
    audioTime_ = sourceTime;
    if (features.active)
        audioSeconds_ += elapsed;
    audio_ = features;
    synthetic_ = synthetic;
}
void PeopleView::setHeightEstimate(double centimetres, double pilotMae, double lower, double upper) {
    if (!std::isfinite(centimetres) || !std::isfinite(pilotMae) || centimetres < 80 || centimetres > 240 ||
        pilotMae < 0) {
        clearHeightEstimate();
        return;
    }
    heightCm_ = centimetres;
    pilotMae_ = pilotMae;
    heightLower_ = std::isfinite(lower) && lower <= centimetres ? lower : 0;
    heightUpper_ = std::isfinite(upper) && upper >= centimetres ? upper : 0;
    updateProportions();
    if (paused_)
        scale_ = heightCm_ / (referenceHumanHeight * 100);
    update();
}
void PeopleView::clearHeightEstimate() {
    heightCm_ = 0;
    pilotMae_ = 0;
    heightLower_ = heightUpper_ = 0;
    updateProportions();
    if (paused_)
        scale_ = 1;
    update();
}
void PeopleView::resetFormation() {
    profileSummary_.clear();
    clearShoulderEstimate();
    audioSeconds_ = 0;
    formation_ = 0;
    activity_ = 0;
    time_ = 0;
    scale_ = 1;
    clearHeightEstimate();
    lastTick_ = clock_.elapsed();
    update();
}
void PeopleView::setShoulderEstimate(double centimetres) {
    shoulderCm_ = std::isfinite(centimetres) && centimetres >= 25 && centimetres <= 65 ? centimetres : 0;
    updateProportions();
}
void PeopleView::clearShoulderEstimate() {
    shoulderCm_ = 0;
    updateProportions();
}
void PeopleView::updateProportions() {
    const double stature = hasHeightEstimate() ? heightCm_ / (referenceHumanHeight * 100) : 1;
    const double factor = shoulderCm_ > 0 ? shoulderCm_ / (referenceHumanShoulders * 100 * stature) : 1;
    if (factor < 0.5 || factor > 1.8) {
        shoulderCm_ = 0;
        updateProportions();
        return;
    }
    if (std::abs(factor - shoulderFactor_) < 1e-9)
        return;
    shoulderFactor_ = factor;
    mesh_ = withShoulderScale(referenceMesh_, factor);
    if (ready_ && context()) {
        std::vector<Vertex> data;
        data.reserve(mesh_.positions.size());
        for (std::size_t i = 0; i < mesh_.positions.size(); ++i) {
            const auto p = mesh_.positions[i], n = mesh_.normals[i];
            data.push_back({float(p.x), float(p.y), float(p.z), float(n.x), float(n.y), float(n.z)});
        }
        makeCurrent();
        vertices_.bind();
        vertices_.write(0, data.data(), int(data.size() * sizeof(Vertex)));
        vertices_.release();
        doneCurrent();
    }
    update();
}
void PeopleView::advance(double seconds) {
    time_ += seconds;
    formation_ = std::min(std::clamp(audioSeconds_ / 5.0, 0.0, 1.0), formation_ + seconds * 0.32);
    const double response = 1 - std::exp(-seconds * 5);
    const double target = audio_.active ? std::clamp((audio_.rmsDb + 55) / 43, 0.0, 1.0) : 0;
    activity_ += (target - activity_) * response;
    const double targetScale = hasHeightEstimate() ? heightCm_ / (referenceHumanHeight * 100) : 1;
    scale_ += (targetScale - scale_) * response;
}
void PeopleView::tick() {
    const auto now = clock_.elapsed();
    const double seconds = std::clamp((now - lastTick_) / 1000.0, 0.0, 0.1);
    lastTick_ = now;
    if (!isVisible())
        return;
    if (!paused_)
        advance(seconds);
    update();
}
void PeopleView::setPaused(bool paused) {
    paused_ = paused;
    lastTick_ = clock_.elapsed();
}
void PeopleView::singleStep() {
    if (paused_) {
        advance(SurfaceSimulation::timeStep);
        update();
    }
}
void PeopleView::setRenderMode(int mode) {
    mode_ = std::clamp(mode, 0, 2);
    update();
}
void PeopleView::setGridVisible(bool visible) {
    gridVisible_ = visible;
    update();
}
void PeopleView::resetCamera() {
    yaw_ = 0.3;
    pitch_ = 0.08;
    distance_ = 3.4;
    pan_ = {};
    update();
}
void PeopleView::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton || event->button() == Qt::RightButton ||
        event->button() == Qt::MiddleButton) {
        drag_ = event->position();
        setFocus();
        setCursor(Qt::ClosedHandCursor);
    }
}
void PeopleView::mouseMoveEvent(QMouseEvent *event) {
    const auto delta = event->position() - drag_;
    drag_ = event->position();
    if (event->buttons() & Qt::LeftButton) {
        yaw_ -= delta.x() * 0.007;
        pitch_ = std::clamp(pitch_ + delta.y() * 0.005, -0.8, 1.1);
    } else if (event->buttons() & (Qt::RightButton | Qt::MiddleButton)) {
        pan_ += QPointF(-delta.x(), delta.y()) * (distance_ * 0.0007);
        pan_.setX(std::clamp(pan_.x(), -1.0, 1.0));
        pan_.setY(std::clamp(pan_.y(), -1.0, 1.0));
    }
    update();
}
void PeopleView::mouseReleaseEvent(QMouseEvent *) {
    setCursor(Qt::OpenHandCursor);
}
void PeopleView::mouseDoubleClickEvent(QMouseEvent *) {
    resetCamera();
}
void PeopleView::wheelEvent(QWheelEvent *event) {
    distance_ = std::clamp(distance_ * std::exp(-event->angleDelta().y() / 120.0 * 0.1), 1.1, 7.0);
    update();
    event->accept();
}
} // namespace sona
