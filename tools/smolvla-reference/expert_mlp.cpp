#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: %s MODEL LAYER INPUT.f32 OUTPUT.f32\n", argv[0]); return 2; }
    std::vector<float> input(50 * 720), output;
    std::ifstream file(argv[3], std::ios::binary);
    file.read(reinterpret_cast<char *>(input.data()), input.size() * sizeof(float));
    if (!file) return 3;
    std::string error;
    auto model = smolvla_expert_mlp::open(argv[1], std::atoi(argv[2]), 4, error);
    if (!model) { std::fprintf(stderr, "%s\n", error.c_str()); return 4; }
    if (!model->run(input.data(), output, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 5; }
    std::ofstream result(argv[4], std::ios::binary);
    result.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
    return result ? 0 : 6;
}
