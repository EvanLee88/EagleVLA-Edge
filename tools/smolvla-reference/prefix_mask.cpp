#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s LANGUAGE_MASK.u8 ATTENTION.u8 POSITIONS.i32\n", argv[0]); return 2; }
    std::vector<uint8_t> language(48), padding, mask;
    std::ifstream file(argv[1], std::ios::binary);
    file.read(reinterpret_cast<char *>(language.data()), language.size());
    if (!file) return 3;
    std::vector<int32_t> positions;
    std::string error;
    if (!smolvla_input::build_prefix_attention({1, 1, 1}, 64, language, padding, mask, positions, error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 4;
    }
    std::ofstream mask_file(argv[2], std::ios::binary), positions_file(argv[3], std::ios::binary);
    mask_file.write(reinterpret_cast<const char *>(mask.data()), mask.size());
    positions_file.write(reinterpret_cast<const char *>(positions.data()), positions.size() * sizeof(int32_t));
    return mask_file && positions_file ? 0 : 5;
}
