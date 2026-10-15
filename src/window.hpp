#pragma once
#include "audio_engine.hpp"
#include <QMainWindow>
#include <memory>

class QAction;
class QComboBox;
class QDockWidget;
class QDoubleSpinBox;
class QLabel;
class QTreeWidget;
class QTabWidget;

namespace sona {
class SurfaceView;
class SpectrumCanvas;
class TraitModels;
class PeopleView;
class Window : public QMainWindow {
  public:
    explicit Window(QWidget *parent = nullptr);
#ifdef SONA_TESTING
    bool runSmokeTest();
#endif
    bool saveSnapshot(const QString &path);
    bool saveMesh(const QString &path);
    void showPeopleView(bool visible);

  private:
    void makeParameters();
    void syncControls();
    void refresh(const Features &features);
    void setPaused(bool paused);
    void importWave();
    void saveImage();
    void saveFeatures();
    void exportMesh();
    void experiment();
    void analyzeVoice();
    void startVoiceAnalysis(bool showResults);
    void loadTraitModels();
    AudioEngine engine_;
    SurfaceView *view_ = nullptr;
    PeopleView *people_ = nullptr;
    QTabWidget *views_ = nullptr;
    std::uint64_t sourceRevision_ = 0;
    double lastAutoAnalysis_ = 0;
    SpectrumCanvas *spectrum_ = nullptr;
    QComboBox *sources_ = nullptr, *renderMode_ = nullptr;
    QAction *pause_ = nullptr, *step_ = nullptr, *microphone_ = nullptr, *listen_ = nullptr;
    QDockWidget *parametersDock_ = nullptr, *spectrumDock_ = nullptr;
    QDockWidget *speakerDock_ = nullptr;
    QTreeWidget *speakerResults_ = nullptr;
    QAction *analyze_ = nullptr;
    std::shared_ptr<TraitModels> traitModels_;
    QDoubleSpinBox *waveSpeed_ = nullptr, *damping_ = nullptr, *restoring_ = nullptr, *gain_ = nullptr;
    QLabel *measurements_ = nullptr, *timing_ = nullptr, *flatness_ = nullptr, *periodicity_ = nullptr;
};
} // namespace sona
