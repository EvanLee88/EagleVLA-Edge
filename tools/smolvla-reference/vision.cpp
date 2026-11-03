#include "jetson_pi_smolvla.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s MMPROJ WIDTH HEIGHT INPUT.rgb OUTPUT.f32\n", argv[0]);
        return 2;
    }
    int width = std::atoi(argv[2]), height = std::atoi(argv[3]);
    if (width <= 0 || height <= 0 || width > 46000 || height > 46000) return 2;
    std::ifstream input(argv[4], std::ios::binary);
    std::vector<uint8_t> rgb((size_t) width * height * 3);
    input.read(reinterpret_cast<char *>(rgb.data()), rgb.size());
    if (!input) return 3;
    std::string error;
    auto vision = smolvla_vision::open(argv[1], 4, error);
    if (!vision) { std::fprintf(stderr, "%s\n", error.c_str()); return 4; }
    std::vector<float> result;
    int tokens = 0, dim = 0;
    if (!vision->encode(rgb.data(), width, height, result, tokens, dim, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 5;
    }
    std::fprintf(stderr, "vision output: %d x %d\n", tokens, dim);
    std::ofstream output(argv[5], std::ios::binary);
    output.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
    return output ? 0 : 6;
}
