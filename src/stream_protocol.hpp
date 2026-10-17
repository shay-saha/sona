#pragma once
#include <QByteArray>
#include <QtEndian>
#include <cmath>
#include <cstring>
#include <istream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sona {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
inline constexpr std::size_t maximumStreamChunk = 4 * 1024 * 1024;
inline QByteArray encodeAudioChunk(const std::vector<float> &samples) {
    if (samples.size() > maximumStreamChunk)
        throw std::invalid_argument("Oversize streaming audio chunk");
    QByteArray packet(qsizetype(4 + samples.size() * 4), Qt::Uninitialized);
    qToLittleEndian<quint32>(quint32(samples.size()), packet.data());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (!std::isfinite(samples[i]) || std::abs(samples[i]) > 1.01f)
            throw std::invalid_argument("Streaming audio must be finite and normalized");
        quint32 bits;
        std::memcpy(&bits, &samples[i], 4);
        qToLittleEndian<quint32>(bits, packet.data() + 4 + i * 4);
    }
    return packet;
}
inline bool readAudioChunk(std::istream &input, std::vector<float> &samples) {
    char header[4];
    input.read(header, 4);
    if (input.gcount() == 0 && input.eof())
        return false;
    if (input.gcount() != 4)
        throw std::invalid_argument("Incomplete streaming audio header");
    const auto count = qFromLittleEndian<quint32>(header);
    if (count > maximumStreamChunk)
        throw std::invalid_argument("Oversize streaming audio chunk");
    QByteArray bytes(qsizetype(count) * 4, Qt::Uninitialized);
    input.read(bytes.data(), bytes.size());
    if (input.gcount() != bytes.size())
        throw std::invalid_argument("Incomplete streaming audio payload");
    samples.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto bits = qFromLittleEndian<quint32>(bytes.constData() + i * 4);
        std::memcpy(&samples[i], &bits, 4);
        if (!std::isfinite(samples[i]) || std::abs(samples[i]) > 1.01f)
            throw std::invalid_argument("Streaming audio must be finite and normalized");
    }
    return true;
}
} // namespace sona
