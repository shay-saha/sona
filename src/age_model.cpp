#include "age_model.hpp"
#include "speaker_features.hpp"
#include <QCryptographicHash>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#ifdef SONA_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace sona {
struct AgeModel::Runtime {
#ifdef SONA_ONNX_RUNTIME
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "sona-age"};
    std::unique_ptr<Ort::Session> session;
    std::string input, output;
#endif
};
AgeModel::AgeModel() : runtime_(std::make_unique<Runtime>()) {}
AgeModel::~AgeModel() = default;
bool AgeModel::runtimeAvailable() {
#ifdef SONA_ONNX_RUNTIME
    return true;
#else
    return false;
#endif
}
std::shared_ptr<AgeModel> AgeModel::load(const QString &graphPath, const QString &calibrationPath) {
#ifndef SONA_ONNX_RUNTIME
    (void)graphPath;
    (void)calibrationPath;
    throw std::runtime_error("Build with ONNX Runtime to enable the age encoder");
#else
    QFile calibration(calibrationPath);
    if (!calibration.open(QIODevice::ReadOnly) || calibration.size() > 64 * 1024)
        throw std::invalid_argument("Unreadable/oversize age calibration");
    QJsonParseError error;
    const auto document = QJsonDocument::fromJson(calibration.readAll(), &error);
    if (error.error != QJsonParseError::NoError || !document.isObject())
        throw std::invalid_argument("Invalid age calibration JSON");
    const auto object = document.object();
    if (object.value("schema_version").toInt() != 1 || object.value("target").toString() != "age_years" ||
        object.value("unit").toString() != "years" || !object.value("passes_gate").isBool())
        throw std::invalid_argument("Unsupported age calibration schema");
    const auto number = [&object](const char *key, double low, double high) {
        const auto value = object.value(QLatin1String(key));
        if (!value.isDouble() || !std::isfinite(value.toDouble()) || value.toDouble() < low ||
            value.toDouble() > high)
            throw std::invalid_argument("Invalid age calibration parameter");
        return value.toDouble();
    };
    auto model = std::shared_ptr<AgeModel>(new AgeModel);
    model->slope_ = number("slope", 0, 5);
    model->intercept_ = number("intercept", -200, 200);
    model->radius_ = number("interval_radius", 0, 100);
    model->mae_ = number("validation_mae", 0, 100);
    model->scope_ = object.value("scope").toString();
    model->supported_ = object.value("passes_gate").toBool();
    if (!model->supported_)
        return model;
    QFile graph(graphPath);
    if (!graph.open(QIODevice::ReadOnly) || graph.size() < 1024 || graph.size() > 512 * 1024 * 1024)
        throw std::invalid_argument("Unreadable/oversize age encoder");
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&graph) || hash.result().toHex() != object.value("model_sha256").toString().toLatin1())
        throw std::invalid_argument("Age encoder checksum mismatch");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
    const auto path = graphPath.toStdWString();
#else
    const auto path = graphPath.toStdString();
#endif
    auto &runtime = *model->runtime_;
    runtime.session = std::make_unique<Ort::Session>(runtime.environment, path.c_str(), options);
    if (runtime.session->GetInputCount() != 1 || runtime.session->GetOutputCount() != 1)
        throw std::invalid_argument("Age encoder must contain exactly one input and one output");
    Ort::AllocatorWithDefaultOptions allocator;
    runtime.input = runtime.session->GetInputNameAllocated(0, allocator).get();
    runtime.output = runtime.session->GetOutputNameAllocated(0, allocator).get();
    if (runtime.input != object.value("input").toString().toStdString() ||
        runtime.output != object.value("output").toString().toStdString() ||
        runtime.output.find("age") == std::string::npos)
        throw std::invalid_argument("Age encoder input/output mismatch");
    const auto type = runtime.session->GetInputTypeInfo(0);
    const auto tensor = type.GetTensorTypeAndShapeInfo();
    const auto shape = tensor.GetShape();
    if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 2 ||
        (shape[0] != -1 && shape[0] != 1) || shape[1] != -1)
        throw std::invalid_argument("Unsupported age encoder input shape");
    return model;
#endif
}
TraitEstimate AgeModel::predict(const std::vector<float> &samples, int rate,
                                const std::vector<double> &features) const {
    TraitEstimate result;
    result.target = "age_years";
    result.unit = "years";
    result.trainingScope = scope_;
    result.pilotMae = mae_;
    result.status = "unsupported_by_pilot";
    if (!supported_ || !runtimeAvailable())
        return result;
    if (features.size() < speakerFeatureNames().size())
        throw std::invalid_argument("Missing age quality descriptors");
    for (double value : features)
        if (!std::isfinite(value))
            throw std::invalid_argument("Non-finite age quality descriptors");
    result.status = "insufficient_clean_periodic_audio";
    if (features[speakerFeature::DurationSeconds] < 2 || features[speakerFeature::ActiveFraction] < 0.3 ||
        features[speakerFeature::PeriodicFraction] < 0.2 || features[speakerFeature::ClippedFraction] > 0.01)
        return result;
#ifdef SONA_ONNX_RUNTIME
    const std::size_t limit = std::size_t(std::max(0, rate)) * 10;
    const auto begin = samples.size() > limit ? samples.end() - std::ptrdiff_t(limit) : samples.begin();
    auto signal = resampleForDescriptor({begin, samples.end()}, rate);
    for (float value : signal)
        if (!std::isfinite(value))
            throw std::invalid_argument("Non-finite age input samples");
    const std::array<int64_t, 2> shape{1, int64_t(signal.size())};
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input =
        Ort::Value::CreateTensor<float>(memory, signal.data(), signal.size(), shape.data(), shape.size());
    const char *inputName = runtime_->input.c_str(), *outputName = runtime_->output.c_str();
    auto outputs = runtime_->session->Run(Ort::RunOptions{nullptr}, &inputName, &input, 1, &outputName, 1);
    if (!outputs[0].IsTensor() || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != 1 ||
        outputs[0].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Invalid age encoder output");
    const double value = double(outputs[0].GetTensorData<float>()[0]) * 100 * slope_ + intercept_;
    if (!std::isfinite(value))
        throw std::runtime_error("Non-finite age estimate");
    result.status = "outside_pilot_feature_range";
    if (value < 20 || value > 74)
        return result;
    result.value = value;
    result.hasInterval = result.available = true;
    result.intervalLower = value - radius_;
    result.intervalUpper = value + radius_;
    result.status = "experimental_pilot_estimate";
#else
    (void)samples;
    (void)rate;
#endif
    return result;
}
} // namespace sona
