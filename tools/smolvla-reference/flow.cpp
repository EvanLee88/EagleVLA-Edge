#include "jetson_pi_smolvla.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s NOISE.f32 VELOCITIES.f32 OUTPUT.f32\n", argv[0]); return 2; }
    constexpr size_t count = 50 * 32;
    std::vector<float> noise(count), velocities(count * 10), result;
    std::ifstream n(argv[1], std::ios::binary), v(argv[2], std::ios::binary);
    n.read(reinterpret_cast<char *>(noise.data()), noise.size() * sizeof(float));
    v.read(reinterpret_cast<char *>(velocities.data()), velocities.size() * sizeof(float));
    if (!n || !v) return 3;
    int step = 0;
    auto callback = [&](const std::vector<float> &, float time, std::vector<float> & out, std::string &) {
        if (step >= 10 || std::abs(time - (1.0f - step * 0.1f)) > 1e-6f) return false;
        out.assign(velocities.begin() + step * count, velocities.begin() + (step + 1) * count);
        ++step;
        return true;
    };
    std::string error;
    if (!smolvla::euler_integrate(noise, 10, callback, result, error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 4;
    }
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char *>(result.data()), result.size() * sizeof(float));
    return out ? 0 : 5;
}
