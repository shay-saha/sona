#pragma once
#include "trait_model.hpp"
#include <memory>

namespace sona {
class AgeModel {
  public:
    static bool runtimeAvailable();
    static std::shared_ptr<AgeModel> load(const QString &graphPath, const QString &calibrationPath);
    ~AgeModel();
    TraitEstimate predict(const std::vector<float> &samples, int sampleRate,
                          const std::vector<double> &features) const;

  private:
    AgeModel();
    struct Runtime;
    std::unique_ptr<Runtime> runtime_;
    QString scope_;
    double slope_ = 1, intercept_ = 0, radius_ = 0, mae_ = 0;
    bool supported_ = false;
};
} // namespace sona
