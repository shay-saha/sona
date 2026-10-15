#include "window.hpp"
#include <QApplication>
#include <QCommandLineParser>
#include <QPalette>
#include <QStyleFactory>
#include <QSurfaceFormat>
#include <QTimer>
#include <iostream>

int main(int argc, char *argv[]) {
    QSurfaceFormat format;
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setAlphaBufferSize(0);
    format.setSamples(4);
    format.setSwapInterval(1);
    QSurfaceFormat::setDefaultFormat(format);
    QApplication app(argc, argv);
    app.setStyle(QStyleFactory::create("Fusion"));
    app.setFont(QFont("Sans Serif", 9));
    QPalette p;
    p.setColor(QPalette::Window, QColor("#292d33"));
    p.setColor(QPalette::WindowText, QColor("#cbd0d7"));
    p.setColor(QPalette::Base, QColor("#1e2227"));
    p.setColor(QPalette::AlternateBase, QColor("#292d33"));
    p.setColor(QPalette::Text, QColor("#cbd0d7"));
    p.setColor(QPalette::Button, QColor("#30353c"));
    p.setColor(QPalette::ButtonText, QColor("#cbd0d7"));
    p.setColor(QPalette::Light, QColor("#484e57"));
    p.setColor(QPalette::Midlight, QColor("#3b4149"));
    p.setColor(QPalette::Mid, QColor("#25292f"));
    p.setColor(QPalette::Dark, QColor("#1a1d21"));
    p.setColor(QPalette::Shadow, QColor("#111317"));
    p.setColor(QPalette::Highlight, QColor("#405f73"));
    p.setColor(QPalette::HighlightedText, Qt::white);
    p.setColor(QPalette::ToolTipBase, QColor("#343a42"));
    p.setColor(QPalette::ToolTipText, QColor("#e0e5eb"));
    p.setColor(QPalette::Disabled, QPalette::Text, QColor("#78818b"));
    p.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#78818b"));
    app.setPalette(p);
    QCoreApplication::setApplicationName("SONA");
    QCoreApplication::setApplicationVersion("0.3.0");
    QCommandLineParser parser;
    parser.setApplicationDescription("C++ audio-driven surface simulation");
    parser.addHelpOption();
    parser.addVersionOption();
    QCommandLineOption screenshot("screenshot", "Save a screenshot and exit.", "path");
#ifdef SONA_TESTING
    QCommandLineOption smoke("smoke-test",
                             "Exercise both OpenGL views, native controls, analysis, and exports.");
    parser.addOption(smoke);
#endif
    QCommandLineOption settle("settle-ms", "Screenshot delay (100–60000 ms).", "milliseconds", "1500");
    QCommandLineOption scene("view", "Initial view: blob or people.", "name", "blob");
    parser.addOptions({screenshot, settle, scene});
    parser.process(app);
    bool valid = false;
    const int delay = parser.value(settle).toInt(&valid);
    if (!valid || delay < 100 || delay > 60000) {
        std::cerr << "--settle-ms must be between 100 and 60000\n";
        return 2;
    }
    if (parser.value(scene) != "blob" && parser.value(scene) != "people") {
        std::cerr << "--view must be blob or people\n";
        return 2;
    }
    sona::Window window;
    window.show();
    window.showPeopleView(parser.value(scene) == "people");
#ifdef SONA_TESTING
    if (parser.isSet(smoke)) {
        QTimer::singleShot(1000, &app, [&] {
            const bool ok = window.runSmokeTest();
            std::cout << (ok ? "GUI / OpenGL checks passed\n" : "GUI / OpenGL checks failed\n");
            app.exit(ok ? 0 : 1);
        });
        return app.exec();
    }
#endif
    if (parser.isSet(screenshot))
        QTimer::singleShot(delay, &app, [&] {
            const bool ok = window.saveSnapshot(parser.value(screenshot));
            std::cout << (ok ? "Screenshot saved\n" : "Screenshot failed\n");
            app.exit(ok ? 0 : 1);
        });
    return app.exec();
}
