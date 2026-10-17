#pragma once
#include "conversation.hpp"
#include <QString>

namespace sona {
DiarizationResult diarizeLocal(const std::vector<float> &samples, const QString &models, int speakers = 0,
                               float threshold = 0.95f, double progressBegin = 0, double progressSpan = 1,
                               bool collectEmbeddings = true);
DiarizationResult diarizeRecording(const std::vector<float> &samples, const QString &models, int speakers = 0,
                                   float threshold = 0.95f, bool collectEmbeddings = true,
                                   double progressBegin = 0, double progressSpan = 1);
} // namespace sona
