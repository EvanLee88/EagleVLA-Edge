#include "jetson_pi_smolvla.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
// Process-local test only: exercise both paths on the same model/cache handle.
struct resident_override {
    const char * name;
    bool was_set;
    std::string previous;
    explicit resident_override(const char * key) : name(key), was_set(std::getenv(key) != nullptr),
        previous(was_set ? std::getenv(key) : "") {}
    bool set(const char * value) const {
#ifdef _WIN32
        return _putenv_s(name, value ? value : "") == 0;
#else
        return value ? setenv(name, value, 1) == 0 : unsetenv(name) == 0;
#endif
    }
    ~resident_override() { set(was_set ? previous.c_str() : nullptr); }
};
bool exact(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
}

// Compare request-local sample caches to the public, uncached forward path.
int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf (skipped without a local model)\n", argv[0]);
        return argc == 1 ? 77 : 2;
    }
    smolvla_backend_options backend;
    std::string error;
    if (!smolvla_backend_options_from_env(backend, error)) return 1;
    auto expert = smolvla_expert::open(argv[1], 4, error, backend);
    auto projection = smolvla_action_input::open(argv[1], 4, error, backend);
    if (!expert || !projection) {
        std::fprintf(stderr, "open: %s\n", error.c_str());
        return 1;
    }
    std::vector<float> noise(50 * 32);
    for (size_t i = 0; i < noise.size(); ++i) noise[i] = std::sin(float(i) * 0.03f);
    std::vector<float> first;
    // Same shape with changed values and masks, shape changes, and a return to A.
    const int tokens[] = {3, 3, 2, 3};
    const int seeds[] = {1, 2, 3, 1};
    for (int test = 0; test < 4; ++test) {
        const size_t n = size_t(tokens[test]);
        const int seed = seeds[test];
        std::vector<std::vector<float>> keys(16, std::vector<float>(n * 320)), values = keys;
        std::vector<uint8_t> mask(n, 1);
        if (seed == 2) std::fill(mask.begin(), mask.end(), uint8_t(0));
        else if (seed == 3) mask[0] = 0;
        for (size_t l = 0; l < keys.size(); ++l) {
            for (size_t i = 0; i < keys[l].size(); ++i) {
                keys[l][i] = 0.03f * std::sin(float(i + l * 17 + seed * 7) * 0.02f);
                values[l][i] = 0.03f * std::cos(float(i + l * 13 + seed * 11) * 0.04f);
            }
        }
        std::vector<float> sampled, reference;
        if (!expert->sample(noise, keys, values, mask, sampled, error)) {
            std::fprintf(stderr, "sample case %d: %s\n", test, error.c_str());
            return 1;
        }
        const bool ok = smolvla::euler_integrate(noise, 10,
            [&](const std::vector<float> & current, float time, std::vector<float> & velocity, std::string & e) {
                std::vector<float> suffix, normalized;
                return projection->encode(current.data(), time, suffix, e) &&
                       expert->forward(suffix.data(), keys, values, mask, normalized, e) &&
                       projection->project_velocity(normalized.data(), velocity, e);
            }, reference, error);
        if (!ok || !exact(sampled, reference)) {
            float max_abs = 0;
            size_t worst = 0, different = 0;
            for (size_t i = 0; i < sampled.size() && i < reference.size(); ++i) {
                if (std::memcmp(&sampled[i], &reference[i], sizeof(float))) ++different;
                const float delta = std::fabs(sampled[i] - reference[i]);
                if (delta > max_abs) { max_abs = delta; worst = i; }
            }
            std::fprintf(stderr, "cached sample differs from uncached Euler case %d: %s; sizes=%zu/%zu different=%zu max_abs=%.9g at %zu\n",
                         test, error.c_str(), sampled.size(), reference.size(), different, max_abs, worst);
            if (worst < sampled.size() && worst < reference.size()) {
                std::fprintf(stderr, "sampled=%.9g (%a), reference=%.9g (%a)\n",
                             sampled[worst], double(sampled[worst]), reference[worst], double(reference[worst]));
            }
            return 1;
        }
        if (backend.backend == smolvla_backend::cuda) {
            resident_override chain("SMOLVLA_EXPERT_CHAIN"), step("SMOLVLA_EXPERT_STEP");
            const char * modes[][2] = {{"0", "0"}, {"1", "0"}, {"1", "1"}, {"1", "0"}};
            for (const auto & mode : modes) {
                std::vector<float> switched;
                if (!chain.set(mode[0]) || !step.set(mode[1]) ||
                    !expert->sample(noise, keys, values, mask, switched, error) ||
                    !exact(switched, reference)) {
                    std::fprintf(stderr, "same-handle Expert chain=%s step=%s case=%d failed: %s\n",
                                 mode[0], mode[1], test, error.c_str());
                    return 1;
                }
            }
        }
        if (test == 0) first = sampled;
        if (test == 3 && !exact(sampled, first)) {
            std::fprintf(stderr, "A->B->shape-change->A failed\n");
            return 1;
        }
        if (test == 3) {
            auto invalid = keys;
            invalid[1][0] = std::numeric_limits<float>::quiet_NaN();
            std::vector<float> failed = sampled;
            if (expert->sample(noise, invalid, values, mask, failed, error) || !failed.empty()) {
                std::fprintf(stderr, "nonfinite prefix rejection failed\n"); return 1;
            }
            invalid = keys;
            std::fill(invalid[1].begin(), invalid[1].end(), std::numeric_limits<float>::max());
            if (expert->sample(noise, invalid, values, mask, failed, error) || !failed.empty()) {
                std::fprintf(stderr, "overflowing finite prefix rejection failed\n"); return 1;
            }
            invalid = keys;
            std::fill(invalid[0].begin(), invalid[0].end(), std::numeric_limits<float>::max());
            if (expert->sample(noise, invalid, values, mask, failed, error) || !failed.empty()) {
                std::fprintf(stderr, "overflowing self-attention rejection failed\n"); return 1;
            }
            invalid = keys;
            invalid[0].pop_back();
            if (expert->sample(noise, invalid, values, mask, failed, error) || !failed.empty()) {
                std::fprintf(stderr, "invalid prefix shape rejection failed\n"); return 1;
            }
            auto bad_mask = mask;
            bad_mask[0] = 2;
            if (expert->sample(noise, keys, values, bad_mask, failed, error) || !failed.empty()) {
                std::fprintf(stderr, "invalid mask rejection failed\n"); return 1;
            }
            auto bad_noise = noise;
            bad_noise[0] = std::numeric_limits<float>::infinity();
            if (expert->sample(bad_noise, keys, values, mask, failed, error) || !failed.empty()) {
                std::fprintf(stderr, "nonfinite noise rejection failed\n"); return 1;
            }
            bad_noise.pop_back();
            if (expert->sample(bad_noise, keys, values, mask, failed, error) || !failed.empty() ||
                !expert->sample(noise, keys, values, mask, sampled, error) || !exact(sampled, first)) {
                std::fprintf(stderr, "invalid noise shape/failure recovery failed: %s\n", error.c_str()); return 1;
            }
            auto aliased = noise;
            if (!expert->sample(aliased, keys, values, mask, aliased, error) || !exact(aliased, first)) {
                std::fprintf(stderr, "input/output aliasing failed: %s\n", error.c_str()); return 1;
            }
        }
    }
    std::puts("PASS: cached sample equals uncached Euler bitwise; value/mask/shape changes and failure recovery");
    return 0;
}
