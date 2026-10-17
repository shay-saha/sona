#include "conversation_controller.hpp"
#include "people_view.hpp"
#include "speaker_features.hpp"
#include "trait_model.hpp"
#include "visuals.hpp"
#include "window.hpp"
#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QDataStream>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMouseEvent>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QWheelEvent>
#include <cmath>
#include <iostream>

namespace sona {
bool Window::runSmokeTest() {
    auto check = [](bool ok, const char *message) {
        if (!ok)
            std::cerr << message << '\n';
        return ok;
    };
    if (!check(view_->isValid() && view_->rendererReady(), "OpenGL renderer did not initialize"))
        return false;
    if (!check(engine_.features().active && engine_.features().pitchHz > 150 &&
                   engine_.features().pitchHz < 180,
               "Demo analysis failed"))
        return false;
    if (!check(view_->simulation().finite() && view_->simulation().time() > 0, "Simulation did not advance"))
        return false;
    pause_->trigger();
    if (!check(engine_.paused() && view_->paused() && step_->isEnabled(), "Pause failed"))
        return false;
    const double before = view_->simulation().time();
    step_->trigger();
    if (!check(std::abs(view_->simulation().time() - before - SurfaceSimulation::timeStep) < 1e-9,
               "Single step failed"))
        return false;
    view_->resetSimulation();
    if (!check(view_->simulation().time() == 0 && view_->simulation().energy() == 0, "Reset failed"))
        return false;
    pause_->trigger();
    sources_->setCurrentIndex(1);
    if (!check(engine_.profile() == 1, "Source switch failed"))
        return false;
    sources_->setCurrentIndex(0);
    renderMode_->setCurrentIndex(1);
    if (!check(view_->renderMode() == 1, "Render mode failed"))
        return false;
    renderMode_->setCurrentIndex(2);
    parametersDock_->show();
    damping_->setValue(4);
    if (!check(view_->simulation().parameters().damping == 4, "Parameters failed"))
        return false;
    damping_->setValue(2.4);
    parametersDock_->hide();
    spectrumDock_->show();
    QApplication::processEvents();
    spectrumDock_->hide();
    resize(900, 620);
    QApplication::processEvents();
    if (!check(view_->width() > 800 && view_->height() > 450, "Viewport is obscured"))
        return false;
    setPaused(true);
    const auto frame = view_->grabFramebuffer();
    if (!check(!frame.isNull(), "Framebuffer capture failed"))
        return false;
    const auto center = frame.pixelColor(frame.width() / 2, frame.height() / 2),
               corner = frame.pixelColor(0, 0);
    if (!check(center != corner, "OpenGL framebuffer contains only background"))
        return false;
    for (int mode = 0; mode < 3; ++mode) {
        renderMode_->setCurrentIndex(mode);
        view_->setGridVisible(true);
        const auto withGrid = view_->grabFramebuffer();
        for (int y = 0; y < withGrid.height(); ++y)
            for (int x = 0; x < withGrid.width(); ++x)
                if (!check(withGrid.pixelColor(x, y).alpha() == 255,
                           "Surface/grid/wire rendering leaked framebuffer alpha"))
                    return false;
        view_->setGridVisible(false);
        const auto withoutGrid = view_->grabFramebuffer();
        for (int y = withGrid.height() * 45 / 100; y < withGrid.height() * 55 / 100; ++y)
            for (int x = withGrid.width() * 45 / 100; x < withGrid.width() * 55 / 100; ++x)
                if (!check(withGrid.pixel(x, y) == withoutGrid.pixel(x, y),
                           "The background grid is visible through the center of the surface"))
                    return false;
    }
    renderMode_->setCurrentIndex(2);
    view_->setGridVisible(true);
    QTemporaryDir dir;
    if (!check(saveMesh(dir.filePath("surface.obj")) && saveSnapshot(dir.filePath("surface.png")),
               "Export failed"))
        return false;
    if (!check(views_->count() == 3 && views_->tabText(0) == "Blob" && views_->tabText(1) == "People" &&
                   views_->tabText(2) == "Conversation",
               "All three native scene tabs must be available"))
        return false;
    showPeopleView(true);
    QApplication::processEvents();
    if (!check(people_->isValid() && people_->rendererReady() && !view_->isVisible(),
               "People renderer did not initialize in its tab"))
        return false;
    if (!check(people_->paused(), "People view must share the pause state"))
        return false;
    people_->resetFormation();
    Features voice;
    voice.active = true;
    voice.rmsDb = -20;
    voice.centroidHz = 1200;
    people_->setAudio({}, 0, true);
    for (int frame = 1; frame <= 60; ++frame)
        people_->setAudio(voice, frame * 0.1, true);
    if (!check(people_->formation() == 0 && people_->observedSeconds() > 5,
               "Paused formation must retain audio evidence without animating"))
        return false;
    for (int step = 0; step < 800; ++step)
        people_->singleStep();
    if (!check(people_->formation() > 0.999, "Active audio must assemble the human surface"))
        return false;
    people_->setHeightEstimate(190, 5.2);
    if (!check(people_->hasHeightEstimate() &&
                   std::abs(people_->heightScale() - 1.9 / referenceHumanHeight) < 1e-9,
               "Height estimates must update stature while paused"))
        return false;
    people_->clearHeightEstimate();
    for (int mode = 0; mode < 3; ++mode) {
        renderMode_->setCurrentIndex(mode);
        people_->setGridVisible(true);
        const auto withGrid = people_->grabFramebuffer();
        if (!check(!withGrid.isNull(), "People framebuffer is empty"))
            return false;
        for (int y = 0; y < withGrid.height(); ++y)
            for (int x = 0; x < withGrid.width(); ++x)
                if (!check(withGrid.pixelColor(x, y).alpha() == 255,
                           "People rendering leaked framebuffer alpha"))
                    return false;
        people_->setGridVisible(false);
        const auto withoutGrid = people_->grabFramebuffer();
        for (int y = withGrid.height() * 47 / 100; y < withGrid.height() * 53 / 100; ++y)
            for (int x = withGrid.width() * 49 / 100; x < withGrid.width() * 51 / 100; ++x)
                if (!check(withGrid.pixel(x, y) == withoutGrid.pixel(x, y),
                           "Grid is visible through the assembled human torso"))
                    return false;
    }
    renderMode_->setCurrentIndex(0);
    const auto beforeOrbit = people_->grabFramebuffer();
    const QPointF centerPoint(people_->width() * 0.5, people_->height() * 0.5);
    QMouseEvent press(QEvent::MouseButtonPress, centerPoint, centerPoint, Qt::LeftButton, Qt::LeftButton,
                      Qt::NoModifier);
    QMouseEvent move(QEvent::MouseMove, centerPoint + QPointF(80, 10), centerPoint + QPointF(80, 10),
                     Qt::NoButton, Qt::LeftButton, Qt::NoModifier);
    QMouseEvent release(QEvent::MouseButtonRelease, centerPoint, centerPoint, Qt::LeftButton, Qt::NoButton,
                        Qt::NoModifier);
    QApplication::sendEvent(people_, &press);
    QApplication::sendEvent(people_, &move);
    QApplication::sendEvent(people_, &release);
    const auto afterOrbit = people_->grabFramebuffer();
    if (!check(beforeOrbit != afterOrbit, "Orbit drag must change the human view"))
        return false;
    QWheelEvent wheel(centerPoint, centerPoint, {}, {0, 120}, Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase,
                      false);
    QApplication::sendEvent(people_, &wheel);
    if (!check(afterOrbit != people_->grabFramebuffer(), "Wheel zoom must change the human view"))
        return false;
    people_->resetCamera();
    people_->setGridVisible(true);
    if (!check(saveMesh(dir.filePath("person.obj")) && saveSnapshot(dir.filePath("person.png")),
               "People export failed"))
        return false;
    const auto savedModels = traitModels_;
    QJsonArray featureNames, zeros, ones;
    for (const auto &name : extendedSpeakerFeatureNames()) {
        featureNames.append(QString::fromStdString(name));
        zeros.append(0);
        ones.append(1);
    }
    const QJsonObject constantModel{{"target", "height_cm"},
                                    {"unit", "cm"},
                                    {"feature_names", featureNames},
                                    {"means", zeros},
                                    {"scales", ones},
                                    {"weights", zeros},
                                    {"intercept", 190},
                                    {"kind", "ridge"},
                                    {"interval_radius", 12},
                                    {"validation", QJsonObject{{"mae", 5.2}}},
                                    {"passes_pilot_gate", true},
                                    {"trained_on", "GUI arithmetic fixture"}};
    auto shoulderModel = constantModel;
    shoulderModel["target"] = "shoulder_cm";
    shoulderModel["intercept"] = 45;
    shoulderModel["interval_radius"] = 5;
    traitModels_ = std::make_shared<TraitModels>(
        TraitModels::fromJson(QJsonDocument(QJsonObject{{"schema_version", 2},
                                                        {"extractor", "sona_features_v2"},
                                                        {"models", QJsonArray{constantModel, shoulderModel}}})
                                  .toJson()));
    const auto voicePath = dir.filePath("voice-fixture.wav");
    {
        QFile file(voicePath);
        if (!check(file.open(QIODevice::WriteOnly), "Could not write voice fixture"))
            return false;
        QDataStream stream(&file);
        stream.setByteOrder(QDataStream::LittleEndian);
        constexpr int sampleRate = 16000, samples = sampleRate * 4;
        stream.writeRawData("RIFF", 4);
        stream << quint32(36 + samples * 2);
        stream.writeRawData("WAVEfmt ", 8);
        stream << quint32(16) << quint16(1) << quint16(1) << quint32(sampleRate) << quint32(sampleRate * 2)
               << quint16(2) << quint16(16);
        stream.writeRawData("data", 4);
        stream << quint32(samples * 2);
        for (int sample = 0; sample < samples; ++sample)
            stream << qint16(10000 * std::sin(2 * pi * 220 * sample / sampleRate));
    }
    showPeopleView(false);
    QString audioError;
    if (!check(engine_.loadWave(voicePath, audioError), "Could not load voice fixture"))
        return false;
    sources_->addItem("voice-fixture.wav");
    sources_->setCurrentIndex(3);
    setPaused(false);
    auto waitFor = [](const std::function<bool()> &finished, int milliseconds) {
        QEventLoop loop;
        QTimer poll, timeout;
        QObject::connect(&poll, &QTimer::timeout, &loop, [&] {
            if (finished())
                loop.quit();
        });
        QObject::connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
        poll.start(20);
        timeout.setSingleShot(true);
        timeout.start(milliseconds);
        loop.exec();
        return finished();
    };
    if (!check(waitFor([this] { return people_->hasHeightEstimate() && analyze_->isEnabled(); }, 7000),
               "The original Blob view must automatically run the shared height estimator"))
        return false;
    if (!check(std::abs(people_->shoulderEstimateCm() - 45) < 1e-9 &&
                   people_->mesh().positions.size() == people_->mesh().normals.size(),
               "Validated shoulder estimates must update the rendered body"))
        return false;
    startVoiceAnalysis(false);
    engine_.selectDemo(2);
    setPaused(true);
    if (!check(waitFor([this] { return analyze_->isEnabled(); }, 3000), "Pending analysis did not finish"))
        return false;
    if (!check(!people_->hasHeightEstimate() && people_->shoulderEstimateCm() == 0 &&
                   people_->formation() == 0,
               "A stale asynchronous analysis must not change the new source's human profile"))
        return false;
    traitModels_ = savedModels;
    showConversationView();
    setPaused(true);
    QApplication::processEvents();
    if (!check(conversation_->isValid() && conversation_->rendererReady() && conversation_->paused(),
               "Conversation tab must initialize and share the pause state"))
        return false;
    const std::vector<SpeechSegment> turns{{0, 0, 5}, {1, 4, 8}, {2, 9, 14}, {0, 14.3, 18}, {1, 17, 22}};
    conversation_->setConversationData(turns, 24);
    conversation_->setConversationTime(19, true);
    conversation_->setConversationStatus("Rendering fixture · three anonymous voices");
    if (!check(conversation_->conversationSummary().speakers.size() == 3 &&
                   conversation_->conversationPositions().size() == 3 &&
                   conversation_->conversationSummary().overlap == 2,
               "Separate figures must reflect cumulative speech and overlapping turns"))
        return false;
    conversation_->setConversationData(turns, 22);
    conversation_->setConversationTime(22, false);
    conversation_->setConversationActivity({0.1f, 0.9f, 0.05f}, 22);
    if (!check(conversation_->conversationSummary().speakers[1].active &&
                   conversation_->conversationSummary().speakers[1].activity > 0.8,
               "Neural voice activity must reach the corresponding 3D figure"))
        return false;
    conversation_->setConversationTime(30, false);
    for (const auto &person : conversation_->conversationSummary().speakers)
        if (!check(!person.active, "Old live evidence must not leave a speaker marked active"))
            return false;
    conversation_->setConversationData(turns, 24);
    conversation_->setConversationActivity({}, 0);
    conversation_->setConversationTime(19, true);
    for (int mode = 0; mode < 3; ++mode) {
        conversation_->setRenderMode(mode);
        const auto frame = conversation_->grabFramebuffer();
        for (int y = 0; y < frame.height(); y += 3)
            for (int x = 0; x < frame.width(); x += 3)
                if (!check(frame.pixelColor(x, y).alpha() == 255,
                           "Conversation rendering must retain opaque figure surfaces"))
                    return false;
    }
    const auto sceneBefore = conversation_->grabFramebuffer();
    QApplication::sendEvent(conversation_, &press);
    QApplication::sendEvent(conversation_, &move);
    QApplication::sendEvent(conversation_, &release);
    if (!check(sceneBefore != conversation_->grabFramebuffer(), "Conversation orbit must move the 3D scene"))
        return false;
    double sought = -1;
    const auto savedSeek = conversation_->onSeek;
    conversation_->onSeek = [&](double seconds) { sought = seconds; };
    const QPointF timelinePoint(conversation_->width() * 0.5, conversation_->height() - 50);
    QMouseEvent timelineClick(QEvent::MouseButtonPress, timelinePoint, timelinePoint, Qt::LeftButton,
                              Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(conversation_, &timelineClick);
    conversation_->onSeek = savedSeek;
    if (!check(sought >= 0 && sought <= 24 && saveMesh(dir.filePath("conversation.obj")) &&
                   saveSnapshot(dir.filePath("conversation.png")),
               "Conversation timeline seeking and exports failed"))
        return false;
    engine_.selectDemo(0);
    if (!check(conversation_->conversationSummary().speakers.empty(),
               "Source changes must remove previous conversation figures"))
        return false;
    setPaused(true);
    for (int switchView = 0; switchView < 4; ++switchView) {
        showPeopleView(switchView % 2 == 0);
        QApplication::processEvents();
    }
    if (!check(view_->rendererReady() && people_->rendererReady(), "Tab switching lost an OpenGL context"))
        return false;
    renderMode_->setCurrentIndex(2);
    resize(1280, 850);
    return true;
}
} // namespace sona
