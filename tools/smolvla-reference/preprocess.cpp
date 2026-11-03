#include "jetson_pi_smolvla.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s WIDTH HEIGHT INPUT.rgb OUTPUT.f32\n", argv[0]);
        return 2;
    }
    const int width = std::atoi(argv[1]);
    const int height = std::atoi(argv[2]);
    if (width <= 0 || height <= 0 || width > 46000 || height > 46000) return 2;
    std::vector<uint8_t> rgb((size_t) width * height * 3);
    std::ifstream input(argv[3], std::ios::binary);
    input.read(reinterpret_cast<char *>(rgb.data()), rgb.size());
    if (!input) return 3;
    std::vector<float> output;
    std::string error;
    if (!smolvla_input::preprocess_rgb(rgb.data(), width, height, 512, 512, output, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 4;
    }
    std::ofstream file(argv[4], std::ios::binary);
    file.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
    return file ? 0 : 5;
}
