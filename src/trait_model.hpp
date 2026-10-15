#pragma once
#include <QByteArray>
#include <QString>
#include <vector>

namespace sona {
struct TraitEstimate {
    QString target, unit, status, trainingScope;
    double value = 0, pilotMae = 0;
    double intervalLower = 0, intervalUpper = 0;
    bool hasInterval = false;
    bool available = false;
};
class TraitModels {
  public:
    static TraitModels fromJson(const QByteArray &data); // Throws on schema/parameter errors.
    std::vector<TraitEstimate> predict(const std::vector<double> &features) const;
    bool usesExtendedFeatures() const {
        return extended_;
    }
    static constexpr int maximumBytes = 8 * 1024 * 1024;

  private:
    struct Model {
        QString target, unit, scope;
        bool supported = false;
        double intercept = 0, mae = 0, gamma = 0, radius = -1;
        std::vector<double> means, scales, weights;
        std::vector<std::size_t> indices;
        std::vector<std::vector<double>> supportVectors;
        std::vector<double> dual;
    };
    std::vector<Model> models_;
    bool extended_ = false;
};
} // namespace sona
