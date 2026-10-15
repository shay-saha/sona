#pragma once
#include <QByteArray>
#include <QString>
#include <vector>

namespace sona {
struct TraitEstimate {
    QString target, unit, status, trainingScope;
    double value = 0, pilotMae = 0;
    bool available = false;
};
class TraitModels {
  public:
    static TraitModels fromJson(const QByteArray &data); // Throws on schema/parameter errors.
    std::vector<TraitEstimate> predict(const std::vector<double> &features) const;

  private:
    struct Model {
        QString target, unit, scope;
        bool supported = false;
        double intercept = 0, mae = 0;
        std::vector<double> means, scales, weights;
    };
    std::vector<Model> models_;
};
} // namespace sona
