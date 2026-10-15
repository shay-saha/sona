#include "speaker_features.hpp"
#include "trait_model.hpp"
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
QJsonObject model() {
    QJsonArray names, means, scales, weights;
    for (const auto &n : sona::speakerFeatureNames()) {
        names.append(QString::fromStdString(n));
        means.append(0);
        scales.append(1);
        weights.append(0);
    }
    means[4] = 220;
    scales[4] = 40;
    weights[4] = 3.5;
    return {{"target", "height_cm"},     {"unit", "cm"},
            {"feature_names", names},    {"means", means},
            {"scales", scales},          {"weights", weights},
            {"intercept", 166},          {"validation", QJsonObject{{"mae", 7.5}}},
            {"passes_pilot_gate", true}, {"trained_on", "Synthetic arithmetic fixture"}};
}
sona::TraitModels load(QJsonObject object) {
    return sona::TraitModels::fromJson(QJsonDocument(QJsonObject{{"schema_version", 1},
                                                                 {"extractor", "sona_features_v1"},
                                                                 {"models", QJsonArray{object}}})
                                           .toJson());
}
bool rejected(QJsonObject object) {
    try {
        load(object);
        return false;
    } catch (const std::invalid_argument &) {
        return true;
    }
}
} // namespace
int main() {
    try {
        std::vector<double> x(sona::speakerFeatureNames().size());
        x[0] = 5;
        x[1] = 0.8;
        x[2] = 0.7;
        x[4] = 240;
        const auto result = load(model()).predict(x).front();
        check(result.available && std::abs(result.value - 167.75) < 1e-12,
              "Portable standardized ridge arithmetic is incorrect");
        auto unvalidated = model();
        unvalidated["passes_pilot_gate"] = false;
        check(!load(unvalidated).predict(x).front().available,
              "Models that fail the pilot gate must abstain");
        x[0] = 0.2;
        check(!load(model()).predict(x).front().available, "Insufficient audio must abstain");
        x[0] = 5;
        x[4] = 10000;
        check(!load(model()).predict(x).front().available, "Extreme out-of-range features must abstain");
        x[4] = 240;
        auto wrong = model();
        wrong["target"] = "unsupported_target";
        check(rejected(wrong), "Unknown prediction targets must be rejected");
        wrong = model();
        wrong["unit"] = "m";
        check(rejected(wrong), "Unit mismatches must be rejected");
        wrong = model();
        auto scales = wrong["scales"].toArray();
        scales[4] = 0;
        wrong["scales"] = scales;
        check(rejected(wrong), "Zero scale must be rejected");
        wrong = model();
        auto names = wrong["feature_names"].toArray();
        names[0] = "other_schema";
        wrong["feature_names"] = names;
        check(rejected(wrong), "Feature schema mismatches must be rejected");
        std::cout
            << "Native model arithmetic, validation gates, audio eligibility, and schema checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
