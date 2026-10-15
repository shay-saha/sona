#include "window.hpp"
#include "age_model.hpp"
#include "people_view.hpp"
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
#include <QTabWidget>
#include <QTextStream>
#include <QToolBar>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>
#include <cmath>

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
    people_ = new PeopleView;
    views_ = new QTabWidget;
    views_->setObjectName("sceneTabs");
    views_->setDocumentMode(true);
    views_->addTab(view_, "Blob");
    views_->addTab(people_, "People");
    views_->setTabToolTip(
        1, "Audio forms a generic human figure. Only an available height estimate changes stature.");
    setCentralWidget(views_);
    view_->onError = [this](const QString &s) { error(this, s); };
    people_->onError = view_->onError;
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
        if (views_->currentIndex() == 1)
            people_->singleStep();
        else
            view_->singleStep();
        refresh(engine_.features());
    });
    auto *reset = simulation->addAction(style()->standardIcon(QStyle::SP_BrowserReload), "Reset");
    reset->setShortcut(Qt::Key_R);
    connect(reset, &QAction::triggered, this, [this] {
        if (views_->currentIndex() == 1)
            people_->resetFormation();
        else
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
    auto *blobView = viewMenu->addAction("Blob");
    blobView->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_1));
    connect(blobView, &QAction::triggered, this, [this] { showPeopleView(false); });
    auto *peopleView = viewMenu->addAction("People");
    peopleView->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_2));
    connect(peopleView, &QAction::triggered, this, [this] { showPeopleView(true); });
    viewMenu->addSeparator();
    auto *camera = viewMenu->addAction("Reset camera");
    camera->setShortcut(Qt::Key_Home);
    connect(camera, &QAction::triggered, this, [this] {
        if (views_->currentIndex() == 1)
            people_->resetCamera();
        else
            view_->resetCamera();
    });
    auto *grid = viewMenu->addAction("Grid");
    grid->setCheckable(true);
    grid->setChecked(true);
    grid->setShortcut(Qt::Key_G);
    connect(grid, &QAction::toggled, view_, &SurfaceView::setGridVisible);
    connect(grid, &QAction::toggled, people_, &PeopleView::setGridVisible);
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
                                 "fullscreen\nCtrl+1 / Ctrl+2: Blob / People\nPeople: right-drag to pan"
                                 "\nCtrl+Enter: analyze recent voice audio");
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
    connect(renderMode_, &QComboBox::currentIndexChanged, people_, &PeopleView::setRenderMode);
    people_->setRenderMode(renderMode_->currentIndex());
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
    connect(views_, &QTabWidget::currentChanged, this, [this, parametersAction](int index) {
        parametersAction->setEnabled(index == 0);
        if (index == 1)
            parametersDock_->hide();
        refresh(engine_.features());
    });
    engine_.onFrame = [this](const Features &f) {
        if (sourceRevision_ != engine_.sourceRevision()) {
            sourceRevision_ = engine_.sourceRevision();
            lastAutoAnalysis_ = 0;
            people_->resetFormation();
            speakerResults_->clear();
        }
        view_->setAudio(f);
        const bool synthetic = !engine_.live() && engine_.profile() >= 0;
        people_->setAudio(f, engine_.elapsed(), synthetic);
        spectrum_->setFrame(f, engine_.sampleRate());
        refresh(f);
        if (views_->currentIndex() == 1 && !synthetic && f.active && analyze_->isEnabled() &&
            engine_.elapsed() - lastAutoAnalysis_ >= 3) {
            lastAutoAnalysis_ = engine_.elapsed();
            startVoiceAnalysis(false);
        }
    };
    engine_.onStatus = [this](const QString &s) {
        error(this, s);
        syncControls();
    };
    const auto research = QDir(QCoreApplication::applicationDirPath()).filePath("../research");
    const auto runtimeModel = research + "/results/nisp_runtime_models.json";
    const auto defaultModel =
        QFileInfo::exists(runtimeModel) ? runtimeModel : research + "/results/nisp_pilot_models.json";
    QFile modelFile(defaultModel);
    if (modelFile.open(QIODevice::ReadOnly) && modelFile.size() <= TraitModels::maximumBytes) {
        try {
            traitModels_ = std::make_shared<TraitModels>(TraitModels::fromJson(modelFile.readAll()));
        } catch (const std::exception &e) {
            error(this, QString::fromUtf8(e.what()));
        }
    }
    if (AgeModel::runtimeAvailable() && QFileInfo::exists(research + "/results/age_calibration.json") &&
        QFileInfo::exists(research + "/cache/age_encoder/age_only.onnx")) {
        try {
            ageModel_ = AgeModel::load(research + "/cache/age_encoder/age_only.onnx",
                                       research + "/results/age_calibration.json");
        } catch (const std::exception &e) {
            error(this, QString::fromUtf8(e.what()));
        }
    }
    syncControls();
}
void Window::showPeopleView(bool visible) {
    views_->setCurrentIndex(visible ? 1 : 0);
    refresh(engine_.features());
}
void Window::loadTraitModels() {
    const auto path =
        QFileDialog::getOpenFileName(this, "Load experimental trait models", {}, "Model JSON (*.json)");
    if (path.isEmpty())
        return;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly) || file.size() > TraitModels::maximumBytes) {
        error(this, "Unreadable model or file exceeds 8 MB.");
        return;
    }
    try {
        traitModels_ = std::make_shared<TraitModels>(TraitModels::fromJson(file.readAll()));
        people_->clearHeightEstimate();
        people_->clearShoulderEstimate();
        people_->setProfileSummary({});
        lastAutoAnalysis_ = 0;
    } catch (const std::exception &e) {
        error(this, QString::fromUtf8(e.what()));
    }
}
void Window::analyzeVoice() {
    startVoiceAnalysis(true);
}
void Window::startVoiceAnalysis(bool showResults) {
    if (!analyze_->isEnabled())
        return;
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
    const auto revision = engine_.sourceRevision();
    auto bank = traitModels_;
    auto ageModel = ageModel_;
    analyze_->setEnabled(false);
    if (showResults)
        statusBar()->showMessage("Analyzing recent voice audio…");
    auto *watcher = new QFutureWatcher<VoiceAnalysis>(this);
    connect(
        watcher, &QFutureWatcher<VoiceAnalysis>::finished, this,
        [this, watcher, source, revision, showResults, bank, ageModel] {
            const auto result = watcher->result();
            watcher->deleteLater();
            analyze_->setEnabled(true);
            if (showResults)
                statusBar()->clearMessage();
            if (revision != engine_.sourceRevision() || bank != traitModels_ || ageModel != ageModel_)
                return;
            if (!result.error.isEmpty()) {
                error(this, result.error);
                return;
            }
            people_->clearHeightEstimate();
            people_->clearShoulderEstimate();
            for (const auto &estimate : result.traits)
                if (estimate.target == "height_cm" && estimate.available)
                    people_->setHeightEstimate(estimate.value, estimate.pilotMae,
                                               estimate.hasInterval ? estimate.intervalLower : 0,
                                               estimate.hasInterval ? estimate.intervalUpper : 0);
            for (const auto &estimate : result.traits)
                if (estimate.target == "shoulder_cm" && estimate.available)
                    people_->setShoulderEstimate(estimate.value);
            QStringList summary;
            for (const auto &estimate : result.traits)
                if (estimate.available && estimate.target != "height_cm") {
                    const QString label = estimate.target == "age_years"   ? "Adult age"
                                          : estimate.target == "weight_kg" ? "Weight"
                                                                           : "Shoulders";
                    summary << label + " " + num(estimate.value, 0) + " " + estimate.unit +
                                   (estimate.hasInterval ? " [" + num(estimate.intervalLower, 0) + "–" +
                                                               num(estimate.intervalUpper, 0) + "]"
                                                         : QString());
                }
            people_->setProfileSummary(summary.isEmpty() ? QString()
                                                         : summary.join(" · ") + " · Nominal 90% ranges");
            if (!showResults)
                return;
            speakerResults_->clear();
            auto *scope = new QTreeWidgetItem(
                speakerResults_,
                {source, num(result.features[speakerFeature::DurationSeconds]) + " s snapshot"});
            scope->setToolTip(0, "Latest audio in memory. A single-speaker research analysis, not continuous "
                                 "speaker identification.");
            auto *traits = new QTreeWidgetItem(speakerResults_, {"Experimental estimates"});
            traits->setFirstColumnSpanned(true);
            if (result.traits.empty())
                new QTreeWidgetItem(traits, {"Model", "Not loaded"});
            for (const auto &p : result.traits) {
                const QString name = p.target == "height_cm"     ? "Height"
                                     : p.target == "weight_kg"   ? "Weight"
                                     : p.target == "shoulder_cm" ? "Shoulder width"
                                                                 : "Age";
                QString value;
                if (p.available)
                    value = num(p.value) + " " + p.unit;
                else if (p.status == "unsupported_by_pilot")
                    value = "Not validated";
                else if (p.status == "outside_pilot_feature_range")
                    value = "Outside pilot range";
                else
                    value = "Insufficient periodic audio";
                auto *item = new QTreeWidgetItem(traits, {name, value});
                item->setToolTip(1, p.trainingScope + "\nHeld-out mean absolute error: " + num(p.pilotMae) +
                                        " " + p.unit + ". This is not an individual confidence interval.");
                if (p.available)
                    new QTreeWidgetItem(item, {"Held-out MAE", num(p.pilotMae) + " " + p.unit});
                if (p.hasInterval) {
                    auto *interval = new QTreeWidgetItem(
                        item, {"90% nominal interval",
                               num(p.intervalLower) + "–" + num(p.intervalUpper) + " " + p.unit});
                    interval->setToolTip(1, "Calibrated on held-out speakers. Coverage is marginal within "
                                            "the research population.");
                    if (p.observedCoverage >= 0)
                        new QTreeWidgetItem(interval,
                                            {"Observed test coverage", num(p.observedCoverage * 100) + "%"});
                }
            }
            const auto &f = result.features;
            const bool hasPitch = f[speakerFeature::MedianPitchHz] > 0;
            auto *measured = new QTreeWidgetItem(speakerResults_, {"Measured voice properties"});
            measured->setFirstColumnSpanned(true);
            new QTreeWidgetItem(
                measured, {"Median pitch", hasPitch ? num(f[speakerFeature::MedianPitchHz]) + " Hz" : "—"});
            new QTreeWidgetItem(measured,
                                {"Pitch P10–P90", hasPitch ? num(f[speakerFeature::PitchP10Hz]) + "–" +
                                                                 num(f[speakerFeature::PitchP90Hz]) + " Hz"
                                                           : "—"});
            new QTreeWidgetItem(
                measured,
                {"Pitch variation",
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
            if (f.size() == extendedSpeakerFeatureNames().size()) {
                for (std::size_t i = 0; i < 4; ++i)
                    new QTreeWidgetItem(measured,
                                        {QString("F%1 resonance").arg(i + 1),
                                         f[resonanceFeature::firstFormantMedian + 2 * i] > 0
                                             ? num(f[resonanceFeature::firstFormantMedian + 2 * i], 0) + " Hz"
                                             : "—"});
                new QTreeWidgetItem(measured,
                                    {"Resonance frames",
                                     num(f[resonanceFeature::validFraction] * 100) + "% of periodic frames"});
            }
            speakerResults_->expandAll();
            speakerDock_->show();
            resizeDocks({speakerDock_}, {460}, Qt::Horizontal);
        });
    watcher->setFuture(QtConcurrent::run([audio = std::move(audio), rate, bank, ageModel] {
        VoiceAnalysis result;
        try {
            result.features = extractExtendedSpeakerFeatures(audio, rate);
            if (bank) {
                auto input = result.features;
                if (!bank->usesExtendedFeatures())
                    input.resize(speakerFeatureNames().size());
                result.traits = bank->predict(input);
            }
            if (ageModel) {
                result.traits.erase(std::remove_if(result.traits.begin(), result.traits.end(),
                                                   [](const auto &p) { return p.target == "age_years"; }),
                                    result.traits.end());
                result.traits.push_back(ageModel->predict(audio, rate, result.features));
            }
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
    people_->setPaused(p);
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
    people_->setPaused(engine_.paused());
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
    timing_->setText(views_->currentIndex() == 1
                         ? QString("%1 fps  |  %2 vertices ")
                               .arg(num(people_->fps(), 0))
                               .arg(people_->mesh().positions.size())
                         : QString("%1 s  |  %2 fps  |  240 Hz ")
                               .arg(num(view_->simulation().time()), num(view_->fps(), 0)));
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
    auto *active = views_->currentIndex() == 1 ? static_cast<QOpenGLWidget *>(people_) : view_;
    if (!active->isValid() ||
        (views_->currentIndex() == 1 ? !people_->rendererReady() : !view_->rendererReady()))
        return false;
    auto image = grab();
    QPainter painter(&image);
    painter.drawImage(QRect(active->mapTo(this, QPoint(0, 0)), active->size()), active->grabFramebuffer());
    painter.end();
    return image.save(path, "PNG");
}
void Window::saveImage() {
    auto path = QFileDialog::getSaveFileName(
        this, "Save screenshot", views_->currentIndex() == 1 ? "person.png" : "surface.png", "PNG (*.png)");
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
    const QJsonObject data{
        {"source", engine_.sourceName()},
        {"view", views_->currentIndex() == 1 ? "people" : "blob"},
        {"sample_rate_hz", engine_.sampleRate()},
        {"pitch_hz", f.pitchHz > 0 ? QJsonValue(f.pitchHz) : QJsonValue(QJsonValue::Null)},
        {"centroid_hz", f.centroidHz},
        {"rms_dbfs", f.rmsDb},
        {"flatness", f.flatness},
        {"yin_periodicity", f.periodicity},
        {"people",
         QJsonObject{
             {"anatomy", "generic"},
             {"formation", people_->formation()},
             {"active_audio_s", people_->observedSeconds()},
             {"height_estimate_cm", people_->hasHeightEstimate() ? QJsonValue(people_->heightEstimateCm())
                                                                 : QJsonValue(QJsonValue::Null)},
             {"pilot_mae_cm", people_->hasHeightEstimate() ? QJsonValue(people_->pilotMaeCm())
                                                           : QJsonValue(QJsonValue::Null)},
             {"shoulder_cm", people_->shoulderEstimateCm() > 0 ? QJsonValue(people_->shoulderEstimateCm())
                                                               : QJsonValue(QJsonValue::Null)}}},
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
    if (views_->currentIndex() == 1) {
        out << "# SONA generic human; audio-reactive portrait\n";
        for (auto position : people_->mesh().positions) {
            position = position * people_->heightScale();
            out << "v " << position.x << ' ' << position.y << ' ' << position.z << '\n';
        }
        for (auto face : people_->mesh().faces)
            out << "f " << face[0] + 1 << ' ' << face[1] + 1 << ' ' << face[2] + 1 << '\n';
        out.flush();
        return out.status() == QTextStream::Ok && file.commit();
    }
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
    auto path = QFileDialog::getSaveFileName(this, "Export mesh",
                                             views_->currentIndex() == 1 ? "person.obj" : "surface.obj",
                                             "Wavefront OBJ (*.obj)");
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
    people_->setPaused(true);
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
} // namespace sona
