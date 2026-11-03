#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s MODEL INPUT.f32 OUTPUT.f32\n", argv[0]); return 2; }
    std::vector<float> input(50 * 720), output;
    std::ifstream file(argv[2], std::ios::binary);
    file.read(reinterpret_cast<char *>(input.data()), input.size() * sizeof(float));
    if (!file) return 3;
    std::string error;
    auto handle = smolvla_action_input::open(argv[1], 4, error);
    if (!handle) { std::fprintf(stderr, "%s\n", error.c_str()); return 4; }
    if (!handle->normalize_expert(input.data(), output, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 5; }
    std::ofstream result(argv[3], std::ios::binary);
    result.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
    return result ? 0 : 6;
}
