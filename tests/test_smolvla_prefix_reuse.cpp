#include "jetson_pi_smolvla.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {
bool select_mode(const char * mode) {
#ifdef _WIN32
    return _putenv_s("SMOLVLA_PREFIX_RESIDENT", mode) == 0;
#else
    return setenv("SMOLVLA_PREFIX_RESIDENT", mode, 1) == 0;
#endif
}
bool exact(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b, size_t tokens) {
    if (a.size() != 16 || b.size() != 16) return false;
    for (size_t layer = 0; layer < 16; ++layer) {
        if (a[layer].size() != tokens * 320 || a[layer].size() != b[layer].size() ||
            std::memcmp(a[layer].data(), b[layer].data(), a[layer].size() * sizeof(float)) != 0) {
            std::fprintf(stderr, "K/V mismatch layer=%zu tokens=%zu\n", layer, tokens);
            return false;
        }
        for (float v : a[layer]) if (!std::isfinite(v)) return false;
    }
    return true;
}
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf (skipped without local model)\n", argv[0]);
        return argc == 1 ? 77 : 2;
    }
    smolvla_backend_options options;
    std::string error;
    if (!smolvla_backend_options_from_env(options, error)) return 1;
    auto model = smolvla_prefix::open(argv[1], 4, error, options);
    if (!model) { std::fprintf(stderr, "%s\n", error.c_str()); return 1; }
    const size_t sizes[] = {1, 3, 3, 2, 3};
    std::vector<std::vector<float>> saved_k, saved_v;
    for (size_t test = 0; test < 5; ++test) {
        const size_t n = sizes[test];
        std::vector<float> input(n * 960);
        std::vector<uint8_t> mask(n * n, 1);
        std::vector<int32_t> positions(n);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = .03f * std::sin(float(i + (test == 2 ? 91 : 37)) * .013f);
        for (size_t i = 0; i < n; ++i) positions[i] = int32_t(i);
        if (test == 3) for (size_t i = 0; i < n; ++i) mask[i] = 0;
        std::vector<std::vector<float>> staged_k, staged_v, resident_k, resident_v;
        if (!select_mode("0") || !model->forward(input, mask, positions, staged_k, staged_v, error) ||
            !select_mode("1") || !model->forward(input, mask, positions, resident_k, resident_v, error) ||
            !exact(staged_k, resident_k, n) || !exact(staged_v, resident_v, n)) {
            std::fprintf(stderr, "case=%zu: %s\n", test, error.c_str()); return 1;
        }
        if (test == 1) { saved_k = resident_k; saved_v = resident_v; }
        if (test == 4 && (!exact(saved_k, resident_k, n) || !exact(saved_v, resident_v, n))) return 1;
        // A failed request must not poison cached graph inputs on the next call.
        input[0] = std::numeric_limits<float>::quiet_NaN();
        if (model->forward(input, mask, positions, resident_k, resident_v, error)) {
            std::fprintf(stderr, "NaN input accepted\n"); return 1;
        }
    }
    std::puts("PASS: all 16 Prefix K/V staged/resident bitwise, changed content/shape/all-mask, failure recovery");
    return 0;
}
