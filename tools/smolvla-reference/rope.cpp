#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s INPUT.f32 POS.i32 OUTPUT.f32\n", argv[0]); return 2; }
    std::vector<float> input(50 * 15 * 64), output;
    std::vector<int32_t> positions(50);
    std::ifstream a(argv[1], std::ios::binary), b(argv[2], std::ios::binary);
    a.read(reinterpret_cast<char *>(input.data()), input.size() * sizeof(float));
    b.read(reinterpret_cast<char *>(positions.data()), positions.size() * sizeof(int32_t));
    if (!a || !b) return 3;
    std::string error;
    if (!smolvla::expert_rope(input.data(), 50, 15, positions.data(), 4, output, error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 4;
    }
    std::ofstream c(argv[3], std::ios::binary);
    c.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
    return c ? 0 : 5;
}
