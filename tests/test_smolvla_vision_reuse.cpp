#include "jetson_pi_smolvla.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct reuse_override {
    bool was_set;
    std::string previous;
    reuse_override() : was_set(std::getenv("SMOLVLA_VISION_REUSE") != nullptr),
        previous(was_set ? std::getenv("SMOLVLA_VISION_REUSE") : "") {}
    bool set(const char * value) const {
#ifdef _WIN32
        return _putenv_s("SMOLVLA_VISION_REUSE", value ? value : "") == 0;
#else
        return value ? setenv("SMOLVLA_VISION_REUSE", value, 1) == 0 : unsetenv("SMOLVLA_VISION_REUSE") == 0;
#endif
    }
    ~reuse_override() { set(was_set ? previous.c_str() : nullptr); }
};
bool exact(const std::vector<float> & a, const std::vector<float> & b) {
    return !a.empty() && a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
}

// Local model only: exercise changed image contents and raw aspect ratios even
// though preprocessing maps all of them to the same fixed-size Idefics3 graph.
int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MMPROJ.gguf (skipped without local model)\n", argv[0]);
        return argc == 1 ? 77 : 2;
    }
    smolvla_backend_options backend;
    std::string error;
    if (!smolvla_backend_options_from_env(backend, error)) return 1;
    auto vision = smolvla_vision::open(argv[1], 4, error, backend);
    auto staged = smolvla_vision::open(argv[1], 4, error, backend);
    if (!vision || !staged) { std::fprintf(stderr, "open: %s\n", error.c_str()); return 1; }
    reuse_override mode;
    const int widths[] = {256, 256, 320, 1, 256};
    const int heights[] = {256, 256, 192, 1, 256};
    const int seeds[] = {1, 2, 3, 4, 1};
    std::vector<float> first;
    for (int c = 0; c < 5; ++c) {
        std::vector<uint8_t> pixels(size_t(widths[c]) * heights[c] * 3);
        for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = uint8_t((i * (7 + seeds[c]) + i / 193 + seeds[c] * 31) % 256);
        std::vector<float> reference;
        int tokens = 0, embedding = 0;
        if (!mode.set("0") || !staged->encode(pixels.data(), widths[c], heights[c], reference, tokens, embedding, error) ||
            tokens != 64 || embedding != 960 || reference.size() != 64 * 960 ||
            !std::all_of(reference.begin(), reference.end(), [](float v) { return std::isfinite(v); })) {
            std::fprintf(stderr, "staged vision case=%d failed: %s\n", c, error.c_str()); return 1;
        }
        for (const char * value : {"1", "1", "0", "1"}) {
            std::vector<float> output;
            int n = 0, d = 0;
            if (!mode.set(value) || !vision->encode(pixels.data(), widths[c], heights[c], output, n, d, error) ||
                n != tokens || d != embedding || !exact(reference, output)) {
                std::fprintf(stderr, "vision case=%d reuse=%s differs: %s\n", c, value, error.c_str()); return 1;
            }
        }
        if (c == 0) first = reference;
        if (c == 4 && !exact(reference, first)) {
            std::fprintf(stderr, "vision A->B->aspect-change->A failed\n"); return 1;
        }
        std::vector<float> failed = reference;
        int n = 7, d = 7;
        if (vision->encode(pixels.data(), 0, heights[c], failed, n, d, error) || !failed.empty() || n != 0 || d != 0) {
            std::fprintf(stderr, "invalid vision input did not clear embeddings\n"); return 1;
        }
        if (!vision->encode(pixels.data(), widths[c], heights[c], failed, n, d, error) || !exact(failed, reference)) {
            std::fprintf(stderr, "vision failure recovery case=%d failed: %s\n", c, error.c_str()); return 1;
        }
    }
    std::puts("PASS: vision cached/staged embeddings bitwise; content/aspect changes, mode switching and failure recovery");
    return 0;
}
