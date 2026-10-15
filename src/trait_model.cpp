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
    if (data.size() > maximumBytes)
        throw std::invalid_argument("Model exceeds 8 MB");
    QJsonParseError error;
    const auto doc = QJsonDocument::fromJson(data, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject())
        throw std::invalid_argument("Invalid model JSON");
    const auto root = doc.object();
    const bool extended =
        root.value("schema_version").toInt() == 2 && root.value("extractor").toString() == "sona_features_v2";
    if (!extended && (root.value("schema_version").toInt() != 1 ||
                      root.value("extractor").toString() != "sona_features_v1"))
        throw std::invalid_argument("Unsupported model schema");
    const auto rows = root.value("models").toArray();
    if (rows.isEmpty() || rows.size() > 4)
        throw std::invalid_argument("Expected 1–4 trait models");
    TraitModels bank;
    bank.extended_ = extended;
    std::set<QString> seen;
    const auto &names = extended ? extendedSpeakerFeatureNames() : speakerFeatureNames();
    for (const auto &row : rows) {
        if (!row.isObject())
            throw std::invalid_argument("Invalid model entry");
        const auto obj = row.toObject();
        Model m;
        m.target = obj.value("target").toString();
        m.unit = obj.value("unit").toString();
        if (!((m.target == "height_cm" && m.unit == "cm") || (m.target == "weight_kg" && m.unit == "kg") ||
              (m.target == "age_years" && m.unit == "years") ||
              (extended && m.target == "shoulder_cm" && m.unit == "cm")))
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
        const QString kind = extended ? obj.value("kind").toString() : "ridge";
        if (kind == "ridge") {
            m.weights = vector(obj.value("weights"), names.size());
            for (std::size_t i = 0; i < names.size(); ++i)
                if (m.weights[i] != 0)
                    m.indices.push_back(i);
        } else if (kind == "rbf_svr") {
            const auto indices = obj.value("feature_indices").toArray();
            std::set<std::size_t> used;
            if (indices.isEmpty() || indices.size() > int(names.size()))
                throw std::invalid_argument("Invalid support-vector feature selection");
            for (const auto &v : indices) {
                const double number = scalar(v);
                if (number < 0 || number >= double(names.size()) || std::floor(number) != number ||
                    !used.insert(std::size_t(number)).second)
                    throw std::invalid_argument("Invalid/duplicate support-vector feature index");
                m.indices.push_back(std::size_t(number));
            }
            const auto vectors = obj.value("support_vectors").toArray();
            if (vectors.isEmpty() || vectors.size() > 4096)
                throw std::invalid_argument("Invalid support-vector count");
            for (const auto &v : vectors)
                m.supportVectors.push_back(vector(v, m.indices.size()));
            m.dual = vector(obj.value("dual_coefficients"), m.supportVectors.size());
            m.gamma = scalar(obj.value("gamma"));
            if (m.gamma <= 0)
                throw std::invalid_argument("Invalid RBF kernel scale");
        } else
            throw std::invalid_argument("Unsupported regression kind");
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
        if (extended) {
            m.radius = scalar(obj.value("interval_radius"));
            if (m.radius < 0)
                throw std::invalid_argument("Invalid prediction interval");
        }
        bank.models_.push_back(std::move(m));
    }
    return bank;
}
std::vector<TraitEstimate> TraitModels::predict(const std::vector<double> &x) const {
    if (x.size() != (extended_ ? extendedSpeakerFeatureNames() : speakerFeatureNames()).size())
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
            std::vector<double> standardized;
            for (std::size_t i : m.indices) {
                const double z = (x[i] - m.means[i]) / m.scales[i];
                standardized.push_back(z);
                if (std::abs(z) > 8)
                    outlier = true;
                if (m.supportVectors.empty())
                    value += z * m.weights[i];
            }
            for (std::size_t s = 0; s < m.supportVectors.size(); ++s) {
                double squaredDistance = 0;
                for (std::size_t j = 0; j < standardized.size(); ++j) {
                    const double delta = standardized[j] - m.supportVectors[s][j];
                    squaredDistance += delta * delta;
                }
                value += m.dual[s] * std::exp(-m.gamma * squaredDistance);
            }
            if (outlier || !std::isfinite(value))
                p.status = "outside_pilot_feature_range";
            else {
                p.available = true;
                p.value = value;
                p.status = "experimental_pilot_estimate";
                if (m.radius >= 0) {
                    p.hasInterval = true;
                    p.intervalLower = value - m.radius;
                    p.intervalUpper = value + m.radius;
                }
            }
        }
        result.push_back(std::move(p));
    }
    return result;
}
} // namespace sona
