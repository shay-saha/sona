#include "window.hpp"
#include "speaker_features.hpp"
#include "trait_model.hpp"
#include "visuals.hpp"
#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QCoreApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QPushButton>
#include <QRandomGenerator>
#include <QSaveFile>
#include <QSignalBlocker>
#include <QStatusBar>
#include <QStyle>
#include <QTemporaryDir>
#include <QTextStream>
#include <QToolBar>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>
#include <cmath>
#include <iostream>

namespace sona {
namespace {
QString num(double n, int digits = 1) {
    return QString::number(n, 'f', digits);
}
void error(QMainWindow *w, const QString &message) {
    w->statusBar()->showMessage(message, 8000);
}
struct VoiceAnalysis {
    std::vector<double> features;
    std::vector<TraitEstimate> traits;
    QString error;
};

class ShapeExperiment : public QDialog {
  public:
    ShapeExperiment(AudioEngine &engine, QWidget *parent) : QDialog(parent), engine_(engine) {
        setWindowTitle("Sound–shape experiment");
        resize(560, 380);
        auto *root = new QVBoxLayout(this);
        auto *row = new QHBoxLayout;
        trial_ = new QLabel;
        row->addWidget(trial_);
        row->addStretch();
        auto *play = new QPushButton("Play stimulus");
        row->addWidget(play);
        root->addLayout(row);
        shapes_ = new QHBoxLayout;
        for (int i = 0; i < 2; ++i) {
            panels_[i] = new QWidget;
            auto *col = new QVBoxLayout(panels_[i]);
            col->addWidget(new ShapeCanvas(i == 1));
            choices_[i] = new QPushButton(i ? "Angular" : "Rounded");
            col->addWidget(choices_[i]);
            connect(choices_[i], &QPushButton::clicked, this, [this, i] {
                if (answered_)
                    return;
                answered_ = true;
                responses_.push_back({number_, profile_, i, leftAngular_ ? 1 : 0});
                for (auto *b : choices_)
                    b->setEnabled(false);
                engine_.stopAudition();
                feedback_->setText(QString("%1 responses recorded").arg(responses_.size()));
            });
        }
        root->addLayout(shapes_, 1);
        feedback_ = new QLabel;
        root->addWidget(feedback_);
        auto *actions = new QHBoxLayout;
        auto *next = new QPushButton("Next");
        auto *csv = new QPushButton("Export CSV…");
        auto *close = new QPushButton("Close");
        actions->addWidget(next);
        actions->addStretch();
        actions->addWidget(csv);
        actions->addWidget(close);
        root->addLayout(actions);
        connect(close, &QPushButton::clicked, this, &QDialog::accept);
        connect(next, &QPushButton::clicked, this, [this] { nextTrial(); });
        connect(play, &QPushButton::clicked, this, [this] {
            QString message;
            if (!engine_.setAudible(true, message)) {
                feedback_->setText(message);
                return;
            }
            if (!answered_) {
                for (auto *b : choices_)
                    b->setEnabled(true);
                feedback_->setText("Choose the shape that matches the sound.");
            }
        });
        connect(csv, &QPushButton::clicked, this, [this] {
            if (responses_.empty()) {
                feedback_->setText("No responses to export.");
                return;
            }
            auto path =
                QFileDialog::getSaveFileName(this, "Export responses", "responses.csv", "CSV (*.csv)");
            if (path.isEmpty())
                return;
            if (!path.endsWith(".csv", Qt::CaseInsensitive))
                path += ".csv";
            QByteArray data("trial,synthetic_stimulus,chosen_shape,angular_shape_position\n");
            for (const auto &r : responses_)
                data += QString("%1,%2,%3,%4\n")
                            .arg(r[0])
                            .arg(r[1] ? "bright" : "warm")
                            .arg(r[2] ? "angular" : "rounded")
                            .arg(r[3] ? "left" : "right")
                            .toUtf8();
            QSaveFile file(path);
            if (!file.open(QIODevice::WriteOnly) || file.write(data) != data.size() || !file.commit())
                feedback_->setText("Could not save responses.");
            else
                feedback_->setText("Saved " + path);
        });
        nextTrial();
    }

  private:
    void nextTrial() {
        ++number_;
        answered_ = false;
        profile_ = QRandomGenerator::global()->bounded(2);
        leftAngular_ = QRandomGenerator::global()->bounded(2) != 0;
        engine_.selectDemo(profile_);
        trial_->setText("Trial " + QString::number(number_));
        feedback_->setText("Play the stimulus before choosing.");
        for (int i = 0; i < 2; ++i) {
            shapes_->removeWidget(panels_[i]);
            choices_[i]->setEnabled(false);
        }
        shapes_->addWidget(panels_[leftAngular_ ? 1 : 0]);
        shapes_->addWidget(panels_[leftAngular_ ? 0 : 1]);
    }
    AudioEngine &engine_;
    QLabel *trial_ = nullptr, *feedback_ = nullptr;
    QHBoxLayout *shapes_ = nullptr;
    QWidget *panels_[2]{};
    QPushButton *choices_[2]{};
    std::vector<std::array<int, 4>> responses_;
    int number_ = 0, profile_ = 0;
    bool answered_ = false, leftAngular_ = false;
};
} // namespace

Window::Window(QWidget *parent) : QMainWindow(parent), engine_(this) {
    setWindowTitle("SONA — Audio analysis");
    resize(1280, 850);
    setMinimumSize(800, 560);
    view_ = new SurfaceView;
    setCentralWidget(view_);
    view_->onError = [this](const QString &s) { error(this, s); };
    auto *file = menuBar()->addMenu("&File");
    auto *open = file->addAction("&Open WAV…");
    open->setShortcut(QKeySequence::Open);
    connect(open, &QAction::triggered, this, &Window::importWave);
    auto *image = file->addAction("Save &screenshot…");
    image->setShortcut(QKeySequence::Save);
    connect(image, &QAction::triggered, this, &Window::saveImage);
    auto *mesh = file->addAction("Export mesh (.obj)…");
    connect(mesh, &QAction::triggered, this, &Window::exportMesh);
    auto *features = file->addAction("Export measurements (.json)…");
    connect(features, &QAction::triggered, this, &Window::saveFeatures);
    file->addSeparator();
    auto *quit = file->addAction("&Quit");
    quit->setShortcut(QKeySequence::Quit);
    connect(quit, &QAction::triggered, this, &QWidget::close);
    auto *simulation = menuBar()->addMenu("&Simulation");
    pause_ = simulation->addAction(style()->standardIcon(QStyle::SP_MediaPause), "Pause");
    pause_->setCheckable(true);
    pause_->setShortcut(Qt::Key_Space);
    connect(pause_, &QAction::triggered, this, [this](bool p) { setPaused(p); });
    step_ = simulation->addAction(style()->standardIcon(QStyle::SP_MediaSkipForward), "Step");
    step_->setShortcut(Qt::Key_N);
    connect(step_, &QAction::triggered, this, [this] {
        view_->singleStep();
        refresh(engine_.features());
    });
    auto *reset = simulation->addAction(style()->standardIcon(QStyle::SP_BrowserReload), "Reset");
    reset->setShortcut(Qt::Key_R);
    connect(reset, &QAction::triggered, this, [this] {
        view_->resetSimulation();
        refresh(engine_.features());
    });
    simulation->addSeparator();
    microphone_ = simulation->addAction("Microphone");
    microphone_->setCheckable(true);
    connect(microphone_, &QAction::triggered, this, [this](bool enabled) {
        QString message;
        if (enabled) {
            if (!engine_.startMicrophone(message))
                error(this, message);
        } else if (sources_->currentIndex() == 3)
            engine_.selectLoadedWave();
        else
            engine_.selectDemo(sources_->currentIndex());
        spectrum_->clear();
        syncControls();
    });
    listen_ = simulation->addAction("Audio output");
    listen_->setCheckable(true);
    connect(listen_, &QAction::triggered, this, [this](bool enabled) {
        QString message;
        if (!engine_.setAudible(enabled, message))
            error(this, message);
        syncControls();
    });
    auto *viewMenu = menuBar()->addMenu("&View");
    auto *camera = viewMenu->addAction("Reset camera");
    camera->setShortcut(Qt::Key_Home);
    connect(camera, &QAction::triggered, view_, &SurfaceView::resetCamera);
    auto *grid = viewMenu->addAction("Grid");
    grid->setCheckable(true);
    grid->setChecked(true);
    grid->setShortcut(Qt::Key_G);
    connect(grid, &QAction::toggled, view_, &SurfaceView::setGridVisible);
    auto *wire = viewMenu->addAction("Cycle rendering");
    wire->setShortcut(Qt::Key_W);
    connect(wire, &QAction::triggered, this,
            [this] { renderMode_->setCurrentIndex((renderMode_->currentIndex() + 1) % 3); });
    auto *full = viewMenu->addAction("Fullscreen");
    full->setShortcut(Qt::Key_F11);
    connect(full, &QAction::triggered, this, [this] { isFullScreen() ? showNormal() : showFullScreen(); });
    auto *tools = menuBar()->addMenu("&Tools");
    analyze_ = tools->addAction("Analyze latest voice audio");
    analyze_->setIconText("Analyze");
    analyze_->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Return));
    connect(analyze_, &QAction::triggered, this, &Window::analyzeVoice);
    auto *loadModel = tools->addAction("Load trait model…");
    connect(loadModel, &QAction::triggered, this, &Window::loadTraitModels);
    tools->addSeparator();
    auto *study = tools->addAction("Sound–shape experiment…");
    connect(study, &QAction::triggered, this, &Window::experiment);
    auto *help = menuBar()->addMenu("&Help");
    auto *controls = help->addAction("Controls");
    connect(controls, &QAction::triggered, this, [this] {
        QMessageBox::information(this, "Controls",
                                 "Drag: orbit\nWheel: zoom\nShift + click: impulse\nSpace: pause / "
                                 "resume\nN: single step\nR: reset surface\nHome: reset camera\nW: cycle "
                                 "rendering\nG: grid\nF2: parameters\nS: spectrum\nF11: "
                                 "fullscreen\nCtrl+Enter: analyze recent voice audio");
    });
    auto *toolbar = addToolBar("Simulation");
    toolbar->setMovable(false);
    toolbar->setIconSize({16, 16});
    toolbar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    sources_ = new QComboBox;
    sources_->setObjectName("sources");
    sources_->setAccessibleName("Audio source");
    sources_->setMinimumWidth(170);
    sources_->addItems({"Warm harmonics", "Bright harmonics", "Rhythmic pulses"});
    toolbar->addWidget(sources_);
    toolbar->addSeparator();
    toolbar->addAction(pause_);
    toolbar->addAction(step_);
    toolbar->addAction(reset);
    toolbar->addSeparator();
    toolbar->addAction(microphone_);
    toolbar->addAction(listen_);
    toolbar->addSeparator();
    renderMode_ = new QComboBox;
    renderMode_->setAccessibleName("Rendering mode");
    renderMode_->addItems({"Surface", "Wireframe", "Surface + wire"});
    renderMode_->setCurrentIndex(2);
    toolbar->addWidget(renderMode_);
    toolbar->addAction(analyze_);
    connect(renderMode_, &QComboBox::currentIndexChanged, view_, &SurfaceView::setRenderMode);
    connect(sources_, &QComboBox::currentIndexChanged, this, [this](int i) {
        if (i < 3)
            engine_.selectDemo(i);
        else
            engine_.selectLoadedWave();
        view_->resetSimulation();
        spectrum_->clear();
        syncControls();
    });
    spectrum_ = new SpectrumCanvas;
    spectrumDock_ = new QDockWidget("Spectrum", this);
    spectrumDock_->setObjectName("spectrumDock");
    spectrumDock_->setWidget(spectrum_);
    spectrumDock_->setAllowedAreas(Qt::BottomDockWidgetArea);
    addDockWidget(Qt::BottomDockWidgetArea, spectrumDock_);
    spectrumDock_->hide();
    auto *spectrumAction = spectrumDock_->toggleViewAction();
    spectrumAction->setShortcut(Qt::Key_S);
    viewMenu->addAction(spectrumAction);
    makeParameters();
    speakerDock_ = new QDockWidget("Speaker analysis", this);
    speakerDock_->setObjectName("speakerAnalysisDock");
    speakerDock_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    speakerResults_ = new QTreeWidget;
    speakerResults_->setHeaderLabels({"Characteristic", "Result"});
    speakerResults_->header()->setSectionResizeMode(0, QHeaderView::Interactive);
    speakerResults_->setColumnWidth(0, 210);
    speakerResults_->header()->setStretchLastSection(true);
    speakerResults_->setMinimumWidth(330);
    speakerDock_->setWidget(speakerResults_);
    addDockWidget(Qt::RightDockWidgetArea, speakerDock_);
    speakerDock_->hide();
    viewMenu->addAction(speakerDock_->toggleViewAction());
    auto *parametersAction = parametersDock_->toggleViewAction();
    parametersAction->setShortcut(Qt::Key_F2);
    viewMenu->addAction(parametersAction);
    auto *stretch = new QWidget;
    stretch->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    toolbar->addWidget(stretch);
    toolbar->addAction(parametersAction);
    measurements_ = new QLabel;
    timing_ = new QLabel;
    QFont font("DejaVu Sans Mono");
    font.setPixelSize(11);
    measurements_->setFont(font);
    timing_->setFont(font);
    statusBar()->addWidget(measurements_, 1);
    statusBar()->addPermanentWidget(timing_);
    statusBar()->setSizeGripEnabled(true);
    engine_.onFrame = [this](const Features &f) {
        view_->setAudio(f);
        spectrum_->setFrame(f, engine_.sampleRate());
        refresh(f);
    };
    engine_.onStatus = [this](const QString &s) {
        error(this, s);
        syncControls();
    };
    const auto defaultModel =
        QDir(QCoreApplication::applicationDirPath()).filePath("../research/results/nisp_pilot_models.json");
    QFile modelFile(defaultModel);
    if (modelFile.open(QIODevice::ReadOnly) && modelFile.size() <= 1024 * 1024) {
        try {
            traitModels_ = std::make_shared<TraitModels>(TraitModels::fromJson(modelFile.readAll()));
        } catch (const std::exception &e) {
            error(this, QString::fromUtf8(e.what()));
        }
    }
    syncControls();
}
void Window::loadTraitModels() {
    const auto path =
        QFileDialog::getOpenFileName(this, "Load experimental trait models", {}, "Model JSON (*.json)");
    if (path.isEmpty())
        return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly) || file.size() > 1024 * 1024) {
        error(this, "Unreadable model or file exceeds 1 MB.");
        return;
    }
    try {
        traitModels_ = std::make_shared<TraitModels>(TraitModels::fromJson(file.readAll()));
    } catch (const std::exception &e) {
        error(this, QString::fromUtf8(e.what()));
    }
}
void Window::analyzeVoice() {
    if (!engine_.live() && engine_.profile() >= 0) {
        error(this, "Select Microphone or open a voice WAV before estimating speaker traits.");
        return;
    }
    auto audio = engine_.recentAudio();
    const int rate = engine_.sampleRate();
    if (audio.size() < std::size_t(rate) * 2) {
        error(this, "Collect at least two seconds of voice audio first.");
        return;
    }
    const QString source = engine_.sourceName();
    auto bank = traitModels_;
    analyze_->setEnabled(false);
    statusBar()->showMessage("Analyzing recent voice audio…");
    auto *watcher = new QFutureWatcher<VoiceAnalysis>(this);
    connect(watcher, &QFutureWatcher<VoiceAnalysis>::finished, this, [this, watcher, source] {
        const auto result = watcher->result();
        watcher->deleteLater();
        analyze_->setEnabled(true);
        statusBar()->clearMessage();
        if (!result.error.isEmpty()) {
            error(this, result.error);
            return;
        }
        speakerResults_->clear();
        auto *scope = new QTreeWidgetItem(
            speakerResults_, {source, num(result.features[speakerFeature::DurationSeconds]) + " s snapshot"});
        scope->setToolTip(0, "Latest audio in memory. A single-speaker research analysis, not continuous "
                             "speaker identification.");
        auto *traits = new QTreeWidgetItem(speakerResults_, {"Experimental estimates"});
        traits->setFirstColumnSpanned(true);
        if (result.traits.empty())
            new QTreeWidgetItem(traits, {"Model", "Not loaded"});
        for (const auto &p : result.traits) {
            QString name = p.target == "height_cm" ? "Height" : (p.target == "weight_kg" ? "Weight" : "Age");
            QString value;
            if (p.available)
                value = num(p.value) + " " + p.unit;
            else if (p.status == "unsupported_by_pilot")
                value = "Unsupported in pilot";
            else if (p.status == "outside_pilot_feature_range")
                value = "Outside pilot range";
            else
                value = "Insufficient periodic audio";
            auto *item = new QTreeWidgetItem(traits, {name, value});
            item->setToolTip(1, p.trainingScope + "\nPilot mean absolute error: " + num(p.pilotMae) + " " +
                                    p.unit + ". This is not an individual confidence interval.");
            if (p.available)
                new QTreeWidgetItem(item, {"Pilot MAE", num(p.pilotMae) + " " + p.unit});
        }
        const auto &f = result.features;
        const bool hasPitch = f[speakerFeature::MedianPitchHz] > 0;
        auto *measured = new QTreeWidgetItem(speakerResults_, {"Measured voice properties"});
        measured->setFirstColumnSpanned(true);
        new QTreeWidgetItem(measured,
                            {"Median pitch", hasPitch ? num(f[speakerFeature::MedianPitchHz]) + " Hz" : "—"});
        new QTreeWidgetItem(measured,
                            {"Pitch P10–P90", hasPitch ? num(f[speakerFeature::PitchP10Hz]) + "–" +
                                                             num(f[speakerFeature::PitchP90Hz]) + " Hz"
                                                       : "—"});
        new QTreeWidgetItem(
            measured, {"Pitch variation",
                       hasPitch ? num(f[speakerFeature::PitchDeviationSemitones]) + " semitones SD" : "—"});
        new QTreeWidgetItem(measured,
                            {"Periodic frames", num(f[speakerFeature::PeriodicFraction] * 100) + "%"});
        auto *harmonicity = new QTreeWidgetItem(
            measured, {"Harmonicity", f[speakerFeature::PeriodicFraction] > 0
                                          ? num(f[speakerFeature::MedianHarmonicityDb]) + " dB"
                                          : "—"});
        harmonicity->setToolTip(0, "Autocorrelation HNR estimate on periodic frames.");
        new QTreeWidgetItem(measured,
                            {"Spectral centroid", num(f[speakerFeature::MeanCentroidHz], 0) + " Hz"});
        new QTreeWidgetItem(measured,
                            {"Level variation", num(f[speakerFeature::LevelDeviationDb]) + " dB SD"});
        new QTreeWidgetItem(measured,
                            {"Input clipping", num(f[speakerFeature::ClippedFraction] * 100, 2) + "%"});
        speakerResults_->expandAll();
        speakerDock_->show();
        resizeDocks({speakerDock_}, {460}, Qt::Horizontal);
    });
    watcher->setFuture(QtConcurrent::run([audio = std::move(audio), rate, bank] {
        VoiceAnalysis result;
        try {
            result.features = extractSpeakerFeatures(audio, rate);
            if (bank)
                result.traits = bank->predict(result.features);
        } catch (const std::exception &e) {
            result.error = QString::fromUtf8(e.what());
        }
        return result;
    }));
}
void Window::makeParameters() {
    parametersDock_ = new QDockWidget("Parameters", this);
    parametersDock_->setObjectName("parametersDock");
    parametersDock_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    auto *widget = new QWidget;
    auto *root = new QVBoxLayout(widget);
    root->setContentsMargins(10, 10, 10, 10);
    auto *surface = new QGroupBox("Surface");
    auto *form = new QFormLayout(surface);
    auto spin = [&](double lo, double hi, double value, double step) {
        auto *s = new QDoubleSpinBox;
        s->setRange(lo, hi);
        s->setValue(value);
        s->setSingleStep(step);
        return s;
    };
    waveSpeed_ = spin(1, 60, 18, 1);
    damping_ = spin(0, 20, 2.4, 0.1);
    restoring_ = spin(1, 100, 14, 1);
    gain_ = spin(0, 3, 1, 0.1);
    form->addRow("Wave coefficient", waveSpeed_);
    form->addRow("Damping", damping_);
    form->addRow("Restoring force", restoring_);
    form->addRow("Input gain", gain_);
    form->addRow("Step", new QLabel("1 / 240 s"));
    form->addRow("Vertices", new QLabel(QString::number(view_->simulation().mesh().directions.size())));
    form->addRow("Triangles", new QLabel(QString::number(view_->simulation().mesh().faces.size())));
    root->addWidget(surface);
    auto *signal = new QGroupBox("Signal");
    auto *data = new QFormLayout(signal);
    flatness_ = new QLabel("—");
    periodicity_ = new QLabel("—");
    data->addRow("Spectral flatness", flatness_);
    data->addRow("YIN periodicity", periodicity_);
    root->addWidget(signal);
    root->addStretch();
    auto update = [this] {
        view_->setParameters({waveSpeed_->value(), damping_->value(), restoring_->value(), gain_->value()});
    };
    for (auto *s : {waveSpeed_, damping_, restoring_, gain_})
        connect(s, &QDoubleSpinBox::valueChanged, this, update);
    parametersDock_->setWidget(widget);
    addDockWidget(Qt::RightDockWidgetArea, parametersDock_);
    parametersDock_->hide();
}
void Window::setPaused(bool p) {
    engine_.setPaused(p);
    view_->setPaused(p);
    syncControls();
}
void Window::syncControls() {
    {
        QSignalBlocker b(pause_);
        pause_->setChecked(engine_.paused());
    }
    pause_->setText(engine_.paused() ? "Resume" : "Pause");
    pause_->setIcon(style()->standardIcon(engine_.paused() ? QStyle::SP_MediaPlay : QStyle::SP_MediaPause));
    step_->setEnabled(engine_.paused());
    view_->setPaused(engine_.paused());
    {
        QSignalBlocker b(microphone_);
        microphone_->setChecked(engine_.live());
    }
    {
        QSignalBlocker b(listen_);
        listen_->setChecked(engine_.audible());
    }
    sources_->setEnabled(!engine_.live());
    listen_->setEnabled(!engine_.live());
    if (!engine_.live()) {
        QSignalBlocker b(sources_);
        sources_->setCurrentIndex(engine_.profile() < 0 ? 3 : engine_.profile());
    }
    refresh(engine_.features());
}
void Window::refresh(const Features &f) {
    measurements_->setText(
        QString(" f0 %1 Hz   |   centroid %2 Hz   |   RMS %3 dBFS")
            .arg(f.pitchHz > 0 ? num(f.pitchHz) : "—", f.active ? num(f.centroidHz, 0) : "—", num(f.rmsDb)));
    timing_->setText(
        QString("%1 s  |  %2 fps  |  240 Hz ").arg(num(view_->simulation().time()), num(view_->fps(), 0)));
    flatness_->setText(f.active ? num(f.flatness, 3) : "—");
    periodicity_->setText(f.pitchHz > 0 ? num(f.periodicity, 3) : "—");
}
void Window::importWave() {
    const auto path = QFileDialog::getOpenFileName(this, "Open WAV", {}, "WAV audio (*.wav *.WAV)");
    if (path.isEmpty())
        return;
    QString message;
    if (!engine_.loadWave(path, message)) {
        error(this, message);
        return;
    }
    {
        QSignalBlocker b(sources_);
        if (sources_->count() > 3)
            sources_->removeItem(3);
        sources_->addItem(QFileInfo(path).fileName());
        sources_->setCurrentIndex(3);
    }
    view_->resetSimulation();
    spectrum_->clear();
    syncControls();
}
bool Window::saveSnapshot(const QString &path) {
    if (!view_->isValid() || !view_->rendererReady())
        return false;
    auto image = grab();
    QPainter painter(&image);
    painter.drawImage(QRect(view_->mapTo(this, QPoint(0, 0)), view_->size()), view_->grabFramebuffer());
    painter.end();
    return image.save(path, "PNG");
}
void Window::saveImage() {
    auto path = QFileDialog::getSaveFileName(this, "Save screenshot", "surface.png", "PNG (*.png)");
    if (path.isEmpty())
        return;
    if (!path.endsWith(".png", Qt::CaseInsensitive))
        path += ".png";
    if (!saveSnapshot(path))
        error(this, "Could not save screenshot.");
}
void Window::saveFeatures() {
    auto path =
        QFileDialog::getSaveFileName(this, "Export measurements", "measurements.json", "JSON (*.json)");
    if (path.isEmpty())
        return;
    if (!path.endsWith(".json", Qt::CaseInsensitive))
        path += ".json";
    const auto &f = engine_.features();
    const auto p = view_->simulation().parameters();
    const QJsonObject data{{"source", engine_.sourceName()},
                           {"sample_rate_hz", engine_.sampleRate()},
                           {"pitch_hz", f.pitchHz > 0 ? QJsonValue(f.pitchHz) : QJsonValue(QJsonValue::Null)},
                           {"centroid_hz", f.centroidHz},
                           {"rms_dbfs", f.rmsDb},
                           {"flatness", f.flatness},
                           {"yin_periodicity", f.periodicity},
                           {"simulation", QJsonObject{{"time_s", view_->simulation().time()},
                                                      {"fixed_step_s", SurfaceSimulation::timeStep},
                                                      {"energy", view_->simulation().energy()},
                                                      {"wave_coefficient", p.waveSpeed},
                                                      {"damping", p.damping},
                                                      {"restoring", p.restoring},
                                                      {"gain", p.gain}}}};
    const auto bytes = QJsonDocument(data).toJson();
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit())
        error(this, "Could not save measurements.");
}
bool Window::saveMesh(const QString &path) {
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly))
        return false;
    QTextStream out(&file);
    out.setRealNumberPrecision(9);
    out << "# SONA surface\n";
    const auto &simulation = view_->simulation();
    for (std::size_t i = 0; i < simulation.mesh().directions.size(); ++i) {
        auto p = simulation.mesh().directions[i] * (1.85 + simulation.displacement()[i]);
        out << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
    }
    for (auto f : simulation.mesh().faces)
        out << "f " << f[0] + 1 << ' ' << f[1] + 1 << ' ' << f[2] + 1 << '\n';
    out.flush();
    return out.status() == QTextStream::Ok && file.commit();
}
void Window::exportMesh() {
    auto path = QFileDialog::getSaveFileName(this, "Export mesh", "surface.obj", "Wavefront OBJ (*.obj)");
    if (path.isEmpty())
        return;
    if (!path.endsWith(".obj", Qt::CaseInsensitive))
        path += ".obj";
    if (!saveMesh(path))
        error(this, "Could not save mesh.");
}
void Window::experiment() {
    const bool live = engine_.live(), paused = engine_.paused();
    const int profile = engine_.profile();
    view_->setPaused(true);
    {
        ShapeExperiment dialog(engine_, this);
        dialog.exec();
    }
    engine_.stopAudition();
    if (profile < 0)
        engine_.selectLoadedWave();
    else
        engine_.selectDemo(profile);
    if (live) {
        QString message;
        if (!engine_.startMicrophone(message))
            error(this, message);
    }
    engine_.setPaused(paused);
    spectrum_->clear();
    syncControls();
}
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
    resize(1280, 850);
    return true;
}
} // namespace sona
