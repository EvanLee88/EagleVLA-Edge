#include "jetson_pi_smolvla.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
bool exact(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool valid(const std::vector<float> & actions) {
    return actions.size() == 300 && std::all_of(actions.begin(), actions.end(),
                                               [](float x) { return std::isfinite(x); });
}

bool cleared(const smolvla_policy_timings & t) {
    return t.vit_ms == 0 && t.prefix_prepare_ms == 0 && t.llm_prefill_ms == 0 &&
           t.action_expert_ms == 0 && t.total_ms == 0;
}
}

// Optional model-backed integration test. No model downloads or robot interfaces.
int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf MMPROJ.gguf (skipped without local models)\n", argv[0]);
        return argc == 1 ? 77 : 2;
    }
    smolvla_backend_options backend;
    std::string error;
    if (!smolvla_backend_options_from_env(backend, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    auto policy = smolvla_policy::open(argv[1], argv[2], 4, error, backend);
    if (!policy) {
        std::fprintf(stderr, "open: %s\n", error.c_str());
        return 1;
    }
    std::vector<uint8_t> pixels_a(256 * 256 * 3), pixels_b(pixels_a.size());
    for (size_t i = 0; i < pixels_a.size(); ++i) {
        pixels_a[i] = uint8_t((i * 13 + i / 257) % 256);
        pixels_b[i] = uint8_t((i * 7 + 91 + i / 193) % 256);
    }
    std::vector<smolvla_rgb_image> images_a(3, {pixels_a.data(), 256, 256});
    std::vector<smolvla_rgb_image> images_b(3, {pixels_b.data(), 256, 256});
    const std::vector<float> state_a = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f};
    const std::vector<float> state_b = {-0.3f, 0.4f, -0.1f, 0.2f, -0.5f, 0.6f};
    std::vector<float> noise(50 * 32);
    for (size_t i = 0; i < noise.size(); ++i) noise[i] = std::sin(float(i) * 0.13f);
    const std::string text_a = "Pick up the red cube.";
    const std::string text_b = "Move the blue object carefully towards the left side of the table, then stop.";
    std::vector<float> a, b, again, fresh_b;
    auto infer = [&](smolvla_policy & model, bool use_b, std::vector<float> & out,
                     smolvla_policy_timings * timings = nullptr) {
        const bool ok = model.infer(use_b ? images_b : images_a, use_b ? text_b : text_a,
                                    use_b ? state_b : state_a, noise, out, error, timings);
        if (!ok || !valid(out)) {
            std::fprintf(stderr, "infer: %s\n", error.c_str());
            return false;
        }
        return true;
    };
    smolvla_policy_timings timings;
    if (!infer(*policy, false, a) || !infer(*policy, true, b) ||
        !infer(*policy, false, again, &timings) || !exact(a, again) || exact(a, b)) {
        std::fprintf(stderr, "A->B->A content/mask reuse failed\n");
        return 1;
    }
    // A second policy verifies that B was not influenced by A's cached values.
    // Exercise public open() with omitted options when no backend/device override
    // is needed; an external caller must inherit the linked library's default.
    auto fresh = backend.backend == smolvla_default_backend() && backend.device == "CUDA0"
        ? smolvla_policy::open(argv[1], argv[2], 4, error)
        : smolvla_policy::open(argv[1], argv[2], 4, error, backend);
    if (!fresh || !infer(*fresh, true, fresh_b) || !exact(b, fresh_b)) {
        std::fprintf(stderr, "fresh B comparison failed: %s\n", error.c_str());
        return 1;
    }
    fresh.reset();
    auto invalid_images = images_a;
    invalid_images[2].width = 0;
    std::vector<float> failed = a;
    timings.vit_ms = timings.prefix_prepare_ms = timings.llm_prefill_ms =
        timings.action_expert_ms = timings.total_ms = 1;
    if (policy->infer(invalid_images, text_a, state_a, noise, failed, error, &timings) ||
        !failed.empty() || !cleared(timings)) {
        std::fprintf(stderr, "late failure did not clear actions/timings\n");
        return 1;
    }
    auto invalid_noise = noise;
    invalid_noise[0] = std::numeric_limits<float>::quiet_NaN();
    if (policy->infer(images_a, text_a, state_a, invalid_noise, failed, error, &timings) ||
        !failed.empty() || !cleared(timings) || !infer(*policy, false, again) || !exact(a, again)) {
        std::fprintf(stderr, "failure recovery failed: %s\n", error.c_str());
        return 1;
    }
    std::puts("PASS: changed image/instruction/state/mask A->B->A, fresh B, timed/untimed parity, failure recovery");
    return 0;
}
