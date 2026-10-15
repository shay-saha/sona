#include "speaker_features.hpp"
#include "trait_model.hpp"
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <iostream>

int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    if (argc != 3) {
        std::cerr << "Usage: sona_traits model.json voice.wav\n";
        return 2;
    }
    try {
        QFile model(QString::fromLocal8Bit(argv[1]));
        if (!model.open(QIODevice::ReadOnly) || model.size() > sona::TraitModels::maximumBytes)
            throw std::runtime_error("Unreadable/oversize model");
        const auto bank = sona::TraitModels::fromJson(model.readAll());
        QFile audio(QString::fromLocal8Bit(argv[2]));
        if (!audio.open(QIODevice::ReadOnly) || audio.size() > 100 * 1024 * 1024)
            throw std::runtime_error("Unreadable/oversize WAV");
        const auto bytes = audio.readAll();
        const auto wave = sona::decodeWave({bytes.begin(), bytes.end()});
        if (!wave)
            throw std::runtime_error(wave.error);
        const auto features = bank.usesExtendedFeatures()
                                  ? sona::extractExtendedSpeakerFeatures(wave.samples, wave.sampleRate)
                                  : sona::extractSpeakerFeatures(wave.samples, wave.sampleRate);
        QJsonArray predictions;
        for (const auto &p : bank.predict(features))
            predictions.append(QJsonObject{
                {"target", p.target},
                {"unit", p.unit},
                {"status", p.status},
                {"estimate", p.available ? QJsonValue(p.value) : QJsonValue(QJsonValue::Null)},
                {"pilot_mae", p.pilotMae},
                {"training_scope", p.trainingScope},
                {"observed_interval_coverage",
                 p.observedCoverage >= 0 ? QJsonValue(p.observedCoverage) : QJsonValue(QJsonValue::Null)},
                {"interval_90_lower",
                 p.hasInterval ? QJsonValue(p.intervalLower) : QJsonValue(QJsonValue::Null)},
                {"interval_90_upper",
                 p.hasInterval ? QJsonValue(p.intervalUpper) : QJsonValue(QJsonValue::Null)}});
        const QJsonObject output{{"input", QFileInfo(audio).fileName()},
                                 {"experimental", true},
                                 {"duration_s", features[sona::speakerFeature::DurationSeconds]},
                                 {"traits", predictions}};
        std::cout << QJsonDocument(output).toJson().toStdString();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
