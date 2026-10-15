#include "speaker_features.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: sona_features file.wav [file.wav ...]\n";
        return 2;
    }
    std::cout << "file";
    for (const auto &name : sona::speakerFeatureNames())
        std::cout << ',' << name;
    std::cout << '\n' << std::setprecision(12);
    for (int i = 1; i < argc; ++i)
        try {
            std::ifstream file(argv[i], std::ios::binary | std::ios::ate);
            if (!file || file.tellg() > 100 * 1024 * 1024)
                throw std::runtime_error("Unreadable file or over 100 MB");
            file.seekg(0);
            const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
            const auto wave = sona::decodeWave(bytes);
            if (!wave)
                throw std::runtime_error(wave.error);
            const auto values = sona::extractSpeakerFeatures(wave.samples, wave.sampleRate);
            std::cout << '"';
            for (const char *p = argv[i]; *p; ++p) {
                if (*p == '"')
                    std::cout << '"';
                std::cout << *p;
            }
            std::cout << '"';
            for (double v : values)
                std::cout << ',' << v;
            std::cout << '\n';
        } catch (const std::exception &e) {
            std::cerr << argv[i] << ": " << e.what() << '\n';
            return 1;
        }
    return 0;
}
