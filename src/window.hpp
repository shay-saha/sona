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

namespace sona {
class SurfaceView;
class SpectrumCanvas;
class TraitModels;
class Window : public QMainWindow {
  public:
    explicit Window(QWidget *parent = nullptr);
    bool runSmokeTest();
    bool saveSnapshot(const QString &path);
    bool saveMesh(const QString &path);

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
    void loadTraitModels();
    AudioEngine engine_;
    SurfaceView *view_ = nullptr;
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
