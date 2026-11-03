#include "jetson_pi_smolvla.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: %s PREFIX.u8 SELF.u8 CROSS.u8 POSITION.i32\n", argv[0]); return 2; }
    std::vector<uint8_t> prefix(241), self, cross;
    std::vector<int32_t> position;
    std::ifstream f(argv[1], std::ios::binary); f.read(reinterpret_cast<char *>(prefix.data()), prefix.size());
    if (!f) return 3;
    std::string error;
    if (!smolvla_input::build_suffix_attention(prefix, 50, self, cross, position, error)) { std::fprintf(stderr, "%s\n", error.c_str()); return 4; }
    std::ofstream a(argv[2], std::ios::binary), b(argv[3], std::ios::binary), c(argv[4], std::ios::binary);
    a.write(reinterpret_cast<const char *>(self.data()), self.size());
    b.write(reinterpret_cast<const char *>(cross.data()), cross.size());
    c.write(reinterpret_cast<const char *>(position.data()), position.size() * sizeof(int32_t));
    return a && b && c ? 0 : 5;
}
