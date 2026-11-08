#include "jetson_pi_smolvla.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

static bool read(const char * path, void * data, size_t size) {
    std::ifstream f(path, std::ios::binary);
    f.read(static_cast<char *>(data), size);
    if (!f) std::fprintf(stderr, "cannot read input: %s\n", path);
    return bool(f);
}

int main(int argc, char ** argv) {
    if (argc != 10) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ IMAGE0 IMAGE1 IMAGE2 STATE NOISE TEXT OUTPUT\n", argv[0]);
        return 2;
    }
    try {
        std::string error;
        smolvla_backend_options backend;
        if (!smolvla_backend_options_from_env(backend, error)) {
            std::fprintf(stderr, "%s\n", error.c_str()); return 2;
        }
        long repeats = 1;
        if (const char * value = std::getenv("SMOLVLA_BENCH_REPEATS")) {
            char * end = nullptr;
            errno = 0;
            repeats = std::strtol(value, &end, 10);
            if (errno || end == value || *end || repeats < 1 || repeats > 10000) {
                std::fprintf(stderr, "SMOLVLA_BENCH_REPEATS must be an integer in [1,10000]\n"); return 2;
            }
        }
        std::vector<std::vector<uint8_t>> rgb(3, std::vector<uint8_t>(256*256*3));
        std::vector<smolvla_rgb_image> images;
        for (int i = 0; i < 3; ++i) {
            if (!read(argv[3+i], rgb[i].data(), rgb[i].size())) return 3;
            images.push_back({rgb[i].data(), 256, 256});
        }
        std::vector<float> state(6), noise(50*32), out;
        if (!read(argv[6], state.data(), state.size()*sizeof(float)) ||
            !read(argv[7], noise.data(), noise.size()*sizeof(float))) return 3;
        using clock = std::chrono::steady_clock;
        const auto load_start = clock::now();
        auto model = smolvla_policy::open(argv[1], argv[2], 4, error, backend);
        const auto load_end = clock::now();
        if (!model) { std::fprintf(stderr, "%s\n", error.c_str()); return 4; }
        std::fprintf(stderr, "policy load_ms=%.3f backend=%s device=%s\n",
                     std::chrono::duration<double, std::milli>(load_end-load_start).count(),
                     backend.backend == smolvla_backend::cuda ? "cuda" : "cpu", backend.device.c_str());
        auto infer = [&](const std::vector<float> & input_noise, std::vector<float> & output,
                         smolvla_policy_timings * timings = nullptr) {
            if (!model->infer(images, argv[8], state, input_noise, output, error, timings)) return false;
            if (output.size() != 50*6 || !std::all_of(output.begin(), output.end(), [](float v) { return std::isfinite(v); })) {
                error = "invalid or non-finite policy output"; return false;
            }
            return true;
        };
        double total_ms = 0;
        double warm_infer_ms = 0;
        smolvla_policy_timings warm_timings;
        for (long repeat = 0; repeat < repeats; ++repeat) {
            smolvla_policy_timings timings;
            const auto start = clock::now();
            if (!infer(noise, out, &timings)) { std::fprintf(stderr, "%s\n", error.c_str()); return 4; }
            const double ms = std::chrono::duration<double, std::milli>(clock::now()-start).count();
            total_ms += ms;
            std::fprintf(stderr, "policy infer repeat=%ld/%ld infer_ms=%.3f vit_ms=%.3f prefix_prepare_ms=%.3f llm_prefill_ms=%.3f action_expert_ms=%.3f total_ms=%.3f\n",
                         repeat+1, repeats, ms, timings.vit_ms, timings.prefix_prepare_ms,
                         timings.llm_prefill_ms, timings.action_expert_ms, timings.total_ms);
            // Exclude the first call only from warm means, not the overall infer mean.
            if (repeat > 0) {
                warm_infer_ms += ms;
                warm_timings.vit_ms += timings.vit_ms;
                warm_timings.prefix_prepare_ms += timings.prefix_prepare_ms;
                warm_timings.llm_prefill_ms += timings.llm_prefill_ms;
                warm_timings.action_expert_ms += timings.action_expert_ms;
                warm_timings.total_ms += timings.total_ms;
            }
        }
        std::fprintf(stderr, "policy infer_mean_ms=%.3f repeats=%ld (same instance)\n", total_ms/repeats, repeats);
        if (repeats > 1) {
            const double warm_count = repeats - 1;
            std::fprintf(stderr, "policy infer_warm_mean repeats=%ld infer_ms=%.3f vit_ms=%.3f prefix_prepare_ms=%.3f llm_prefill_ms=%.3f action_expert_ms=%.3f total_ms=%.3f\n",
                         repeats-1, warm_infer_ms/warm_count, warm_timings.vit_ms/warm_count,
                         warm_timings.prefix_prepare_ms/warm_count, warm_timings.llm_prefill_ms/warm_count,
                         warm_timings.action_expert_ms/warm_count, warm_timings.total_ms/warm_count);
        }
        if (std::getenv("SMOLVLA_TEST_REUSE")) {
            const auto first = out;
            auto alternate_noise = noise;
            for (auto & value : alternate_noise) value = -value;
            std::vector<float> alternate;
            if (!infer(alternate_noise, alternate) || !infer(noise, out) || out != first) {
                std::fprintf(stderr, "policy graph reuse failed: %s\n", error.empty() ? "A-B-A output mismatch" : error.c_str());
                return 4;
            }
            std::fprintf(stderr, "policy graph reuse A-B-A exact repeat: success\n");
        }
        std::ofstream f(argv[9], std::ios::binary);
        f.write(reinterpret_cast<const char *>(out.data()), out.size()*sizeof(float));
        return f ? 0 : 5;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "policy failed: %s\n", e.what()); return 4;
    }
}
