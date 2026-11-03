#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc != 6) { std::fprintf(stderr, "usage: %s MODEL LAYER HIDDEN ATTENTION OUTPUT\n", argv[0]); return 2; }
    std::vector<float> hidden(50*720), attention(50*960), output;
    std::ifstream a(argv[3], std::ios::binary), b(argv[4], std::ios::binary);
    a.read(reinterpret_cast<char *>(hidden.data()), hidden.size()*sizeof(float));
    b.read(reinterpret_cast<char *>(attention.data()), attention.size()*sizeof(float));
    if (!a || !b) return 3;
    std::string error;
    auto model = smolvla_expert_mlp::open(argv[1], std::atoi(argv[2]), 4, error);
    if (!model) { std::fprintf(stderr, "%s\n", error.c_str()); return 4; }
    if (!model->finish_layer(hidden.data(), attention.data(), output, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 5; }
    std::ofstream result(argv[5], std::ios::binary);
    result.write(reinterpret_cast<const char *>(output.data()), output.size()*sizeof(float));
    return result ? 0 : 6;
}
