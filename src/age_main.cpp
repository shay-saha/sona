#include "age_model.hpp"
#include "speaker_features.hpp"
#include <QCoreApplication>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <iostream>

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    if (argc == 2 && QString::fromLocal8Bit(argv[1]) == "--capabilities") {
        std::cout << QJsonDocument(QJsonObject{{"age_runtime", sona::AgeModel::runtimeAvailable()}})
                         .toJson()
                         .toStdString();
        return 0;
    }
    if (argc != 4) {
        std::cerr << "Usage: sona_age age_only.onnx calibration.json voice.wav\n";
        return 2;
    }
    try {
        const auto model =
            sona::AgeModel::load(QString::fromLocal8Bit(argv[1]), QString::fromLocal8Bit(argv[2]));
        QFile audio(QString::fromLocal8Bit(argv[3]));
        if (!audio.open(QIODevice::ReadOnly) || audio.size() > 100 * 1024 * 1024)
            throw std::runtime_error("Unreadable/oversize WAV");
        const auto bytes = audio.readAll();
        const auto wave = sona::decodeWave({bytes.begin(), bytes.end()});
        if (!wave)
            throw std::runtime_error(wave.error);
        const auto features = sona::extractSpeakerFeatures(wave.samples, wave.sampleRate);
        const auto p = model->predict(wave.samples, wave.sampleRate, features);
        const QJsonObject output{
            {"target", p.target},
            {"status", p.status},
            {"estimate", p.available ? QJsonValue(p.value) : QJsonValue(QJsonValue::Null)},
            {"interval_90_lower", p.hasInterval ? QJsonValue(p.intervalLower) : QJsonValue(QJsonValue::Null)},
            {"interval_90_upper", p.hasInterval ? QJsonValue(p.intervalUpper) : QJsonValue(QJsonValue::Null)},
            {"validation_mae", p.pilotMae},
            {"training_scope", p.trainingScope}};
        std::cout << QJsonDocument(output).toJson().toStdString();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
