#include "trait_model.hpp"
#include "speaker_features.hpp"
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <cmath>
#include <set>
#include <stdexcept>

namespace sona {
namespace {
double scalar(const QJsonValue &v) {
    if (!v.isDouble() || !std::isfinite(v.toDouble()))
        throw std::invalid_argument("Non-finite/missing model number");
    return v.toDouble();
}
std::vector<double> vector(const QJsonValue &v, std::size_t count) {
    if (!v.isArray() || std::size_t(v.toArray().size()) != count)
        throw std::invalid_argument("Wrong model vector length");
    std::vector<double> out;
    out.reserve(count);
    for (const auto &x : v.toArray())
        out.push_back(scalar(x));
    return out;
}
} // namespace
TraitModels TraitModels::fromJson(const QByteArray &data) {
    if (data.size() > 1024 * 1024)
        throw std::invalid_argument("Model exceeds 1 MB");
    QJsonParseError error;
    const auto doc = QJsonDocument::fromJson(data, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject())
        throw std::invalid_argument("Invalid model JSON");
    const auto root = doc.object();
    if (root.value("schema_version").toInt() != 1 || root.value("extractor").toString() != "sona_features_v1")
        throw std::invalid_argument("Unsupported model schema");
    const auto rows = root.value("models").toArray();
    if (rows.isEmpty() || rows.size() > 3)
        throw std::invalid_argument("Expected 1–3 trait models");
    TraitModels bank;
    std::set<QString> seen;
    const auto &names = speakerFeatureNames();
    for (const auto &row : rows) {
        if (!row.isObject())
            throw std::invalid_argument("Invalid model entry");
        const auto obj = row.toObject();
        Model m;
        m.target = obj.value("target").toString();
        m.unit = obj.value("unit").toString();
        if (!((m.target == "height_cm" && m.unit == "cm") || (m.target == "weight_kg" && m.unit == "kg") ||
              (m.target == "age_years" && m.unit == "years")))
            throw std::invalid_argument("Unsupported prediction target or units");
        if (!seen.insert(m.target).second)
            throw std::invalid_argument("Duplicate prediction target");
        const auto schema = obj.value("feature_names").toArray();
        if (std::size_t(schema.size()) != names.size())
            throw std::invalid_argument("Feature schema mismatch");
        for (std::size_t i = 0; i < names.size(); ++i)
            if (schema[int(i)].toString() != QString::fromStdString(names[i]))
                throw std::invalid_argument("Feature order mismatch");
        m.means = vector(obj.value("means"), names.size());
        m.scales = vector(obj.value("scales"), names.size());
        m.weights = vector(obj.value("weights"), names.size());
        for (double s : m.scales)
            if (s <= 1e-12)
                throw std::invalid_argument("Invalid model scale");
        if (!obj.value("passes_pilot_gate").isBool())
            throw std::invalid_argument("Missing validation gate");
        m.supported = obj.value("passes_pilot_gate").toBool();
        m.scope = obj.value("trained_on").toString();
        m.intercept = scalar(obj.value("intercept"));
        m.mae = scalar(obj.value("validation").toObject().value("mae"));
        if (m.mae < 0)
            throw std::invalid_argument("Invalid model validation error");
        bank.models_.push_back(std::move(m));
    }
    return bank;
}
std::vector<TraitEstimate> TraitModels::predict(const std::vector<double> &x) const {
    if (x.size() != speakerFeatureNames().size())
        throw std::invalid_argument("Wrong feature count");
    for (double v : x)
        if (!std::isfinite(v))
            throw std::invalid_argument("Invalid audio features");
    std::vector<TraitEstimate> result;
    for (const auto &m : models_) {
        TraitEstimate p;
        p.target = m.target;
        p.unit = m.unit;
        p.pilotMae = m.mae;
        p.trainingScope = m.scope;
        if (!m.supported)
            p.status = "unsupported_by_pilot";
        else if (x[speakerFeature::DurationSeconds] < 2 || x[speakerFeature::ActiveFraction] < 0.3 ||
                 x[speakerFeature::PeriodicFraction] < 0.2 || x[speakerFeature::ClippedFraction] > 0.01)
            p.status = "insufficient_clean_periodic_audio";
        else {
            bool outlier = false;
            double value = m.intercept;
            for (std::size_t i = 0; i < x.size(); ++i)
                if (m.weights[i] != 0) {
                    const double z = (x[i] - m.means[i]) / m.scales[i];
                    if (std::abs(z) > 8)
                        outlier = true;
                    value += z * m.weights[i];
                }
            if (outlier || !std::isfinite(value))
                p.status = "outside_pilot_feature_range";
            else {
                p.available = true;
                p.value = value;
                p.status = "experimental_pilot_estimate";
            }
        }
        result.push_back(std::move(p));
    }
    return result;
}
} // namespace sona
