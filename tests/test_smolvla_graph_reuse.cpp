#include "jetson_pi_smolvla.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr float reuse_tolerance = 1e-6f;
constexpr float cuda_attention_tolerance = 1e-3f;
// CPU forms phases by recurrent FP32 multiplication; CUDA uses powf per
// dimension and fast-math sinf/cosf. At position 252, even dimension zero
// (identical phase) differed by 4.39e-5 due to device trig argument reduction.
// This cross-backend bound is separate from the strict cache-reuse tolerance.
constexpr float cuda_rope_tolerance = 1e-4f;

bool require(bool condition, const std::string & message) {
    if (!condition) std::fprintf(stderr, "%s\n", message.c_str());
    return condition;
}

// Check each operand explicitly: fabs(NaN) > tolerance is false.
bool compare(const std::vector<float> & a, const std::vector<float> & b,
             float tolerance, const std::string & label, float * observed = nullptr) {
    if (!require(a.size() == b.size() && !a.empty(), label + ": output size mismatch/empty output")) return false;
    float max_error = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            std::fprintf(stderr, "%s: non-finite output at %zu (%g, %g)\n", label.c_str(), i, a[i], b[i]);
            return false;
        }
        const float error = std::fabs(a[i] - b[i]);
        if (error > max_error) { max_error = error; worst = i; }
    }
    if (observed) *observed = max_error;
    if (max_error > tolerance) {
        std::fprintf(stderr, "%s: max_abs=%.9g > %.9g at %zu (%g, %g)\n",
                     label.c_str(), max_error, tolerance, worst, a[worst], b[worst]);
        return false;
    }
    return true;
}

struct inputs {
    size_t nq, nk;
    bool bf16;
    std::vector<float> q, k, v;
    std::vector<uint8_t> mask;
    std::vector<int32_t> qpos, kpos;

    inputs(size_t queries, size_t keys, int seed, bool round)
        : nq(queries), nk(keys), bf16(round), q(nq * 15 * 64), k(nk * 5 * 64), v(k.size()),
          mask(nq * nk), qpos(nq), kpos(nk) {
        for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(float(i + seed * 37) * 0.03f);
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = std::cos(float(i + seed * 23) * 0.02f);
            // Include a nonzero, head-dependent mean so an all-zero masked row
            // cannot accidentally satisfy the uniform-attention check.
            v[i] = 0.2f + 0.04f * float((i / 64) % 5) + std::sin(float(i + seed * 11) * 0.01f);
        }
        for (size_t t = 0; t < nq; ++t) {
            qpos[t] = int32_t(t + seed * 7);
            for (size_t s = 0; s < nk; ++s) mask[t * nk + s] = uint8_t((t + s + size_t(seed)) % 4 != 0);
        }
        for (size_t t = 0; t < nk; ++t) kpos[t] = int32_t(t + seed * 5);
        // LeRobot masks with float32 minimum, so row zero must attend uniformly
        // to every value, rather than producing zeros or NaNs.
        std::fill(mask.begin(), mask.begin() + nk, uint8_t(0));
    }
};

struct outputs {
    std::vector<float> attention, qrope, krope;
};

std::string describe(const inputs & in) {
    return "(" + std::to_string(in.nq) + "," + std::to_string(in.nk) +
           ") bf16=" + (in.bf16 ? "true" : "false");
}

bool compute(const inputs & in, const smolvla_backend_options & backend, outputs & out, std::string & error) {
    if (backend.backend == smolvla_default_backend() && backend.device == "CUDA0") {
        // Omitted options must follow the linked build, not assume a CPU default.
        return smolvla::attention(in.q.data(), in.k.data(), in.v.data(), in.nq, in.nk, in.mask.data(),
                                  1, out.attention, error, in.bf16) &&
               smolvla::expert_rope(in.q.data(), in.nq, 15, in.qpos.data(), 1, out.qrope, error) &&
               smolvla::expert_rope(in.k.data(), in.nk, 5, in.kpos.data(), 1, out.krope, error);
    }
    return smolvla::attention(in.q.data(), in.k.data(), in.v.data(), in.nq, in.nk, in.mask.data(),
                              1, out.attention, error, in.bf16, backend) &&
           smolvla::expert_rope(in.q.data(), in.nq, 15, in.qpos.data(), 1, out.qrope, error, backend) &&
           smolvla::expert_rope(in.k.data(), in.nk, 5, in.kpos.data(), 1, out.krope, error, backend);
}

bool same_backend(const outputs & reused, const outputs & fresh, const std::string & label) {
    return compare(reused.attention, fresh.attention, reuse_tolerance, label + " attention") &&
           compare(reused.qrope, fresh.qrope, reuse_tolerance, label + " query RoPE") &&
           compare(reused.krope, fresh.krope, reuse_tolerance, label + " key RoPE");
}

float round_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool masked_row(const inputs & in, const outputs & out, const std::string & label) {
    std::vector<float> expected(15 * 64), actual;
    const float probability = in.bf16 ? round_bf16(1.0f / float(in.nk)) : 1.0f / float(in.nk);
    for (size_t h = 0; h < 15; ++h) {
        for (size_t d = 0; d < 64; ++d) {
            double sum = 0.0;
            for (size_t t = 0; t < in.nk; ++t) sum += double(in.v[(t * 5 + h / 3) * 64 + d]) * probability;
            expected[h * 64 + d] = float(sum);
        }
    }
    if (!require(out.attention.size() == in.nq * 15 * 64, label + ": invalid attention output size")) return false;
    actual.assign(out.attention.begin(), out.attention.begin() + expected.size());
    // Independent analytical semantics check; allow the same fixed absolute
    // bound as CPU/CUDA attention, including CUDA matmul rounding of values.
    return compare(actual, expected, cuda_attention_tolerance, label + " all-masked uniform row");
}

bool reused_and_fresh(const inputs & in, const smolvla_backend_options & backend, outputs & reused) {
    const std::string label = std::string(backend.backend == smolvla_backend::cpu ? "CPU " : "CUDA ") + describe(in);
    std::string error;
    const bool reused_ok = compute(in, backend, reused, error);
    if (!require(reused_ok, label + " reused graph failed: " + error)) return false;
    outputs fresh;
    bool ok = false;
    // The caches are thread-local. A newly created worker has no cached graphs.
    std::thread worker([&] { ok = compute(in, backend, fresh, error); });
    worker.join();
    return require(ok, label + " fresh graph failed: " + error) &&
           same_backend(reused, fresh, label + " reused/fresh") && masked_row(in, reused, label);
}

struct error_summary {
    float attention_f32 = 0.0f, attention_bf16 = 0.0f, rope = 0.0f;
};

bool exercise(const inputs & in, const smolvla_backend_options & selected,
              outputs & cpu, outputs & cuda, error_summary & errors) {
    smolvla_backend_options cpu_backend;
    cpu_backend.backend = smolvla_backend::cpu;
    if (!reused_and_fresh(in, cpu_backend, cpu)) return false;
    if (selected.backend != smolvla_backend::cuda) return true;
    if (!reused_and_fresh(in, selected, cuda)) return false;

    float attention_error = 0.0f, qrope_error = 0.0f, krope_error = 0.0f;
    // Evaluate all comparisons before returning so the actual errors are reported
    // for both F32 and BF16, even when attention exceeds its fixed tolerance.
    const bool attention_ok = compare(cpu.attention, cuda.attention, cuda_attention_tolerance,
                                      "CPU/CUDA attention " + describe(in), &attention_error);
    const bool qrope_ok = compare(cpu.qrope, cuda.qrope, cuda_rope_tolerance,
                                  "CPU/CUDA query RoPE " + describe(in), &qrope_error);
    const bool krope_ok = compare(cpu.krope, cuda.krope, cuda_rope_tolerance,
                                  "CPU/CUDA key RoPE " + describe(in), &krope_error);
    std::fprintf(stderr, "CPU/CUDA %s max_abs: attention=%.9g query_rope=%.9g key_rope=%.9g\n",
                 describe(in).c_str(), attention_error, qrope_error, krope_error);
    float & attention_max = in.bf16 ? errors.attention_bf16 : errors.attention_f32;
    attention_max = std::max(attention_max, attention_error);
    errors.rope = std::max(errors.rope, std::max(qrope_error, krope_error));

    // CPU -> CUDA -> CPU uses identical shapes, thread count and data, and must
    // not accidentally reuse the other backend's scheduler/device allocation.
    outputs cpu_again;
    if (!reused_and_fresh(in, cpu_backend, cpu_again) ||
        !same_backend(cpu, cpu_again, "CPU-CUDA-CPU " + describe(in))) return false;
    return attention_ok && qrope_ok && krope_ok;
}

bool reject_missing_device() {
    smolvla_backend_options invalid;
    invalid.backend = smolvla_backend::cuda;
    invalid.device = "CUDA__smolvla_graph_reuse_nonexistent_device__";
    const inputs in(2, 3, 0, false);
    std::vector<float> output(1, 123.0f);
    std::string error;
    // Run after valid cache entries exist: an incomplete cache key must not let
    // an invalid device silently hit a CPU or another CUDA device's graph.
    const bool attention_ok = smolvla::attention(in.q.data(), in.k.data(), in.v.data(), in.nq, in.nk,
                                                in.mask.data(), 1, output, error, in.bf16, invalid);
    if (!require(!attention_ok && !error.empty() && output.empty(),
                 "nonexistent CUDA device must strictly reject attention without CPU fallback")) return false;
    output.assign(1, 123.0f);
    error.clear();
    const bool rope_ok = smolvla::expert_rope(in.q.data(), in.nq, 15, in.qpos.data(), 1, output, error, invalid);
    return require(!rope_ok && !error.empty() && output.empty(),
                   "nonexistent CUDA device must strictly reject RoPE without CPU fallback");
}
} // namespace

int main() {
    smolvla_backend_options selected;
    std::string error;
    if (!require(selected.backend == smolvla_default_backend(), "public backend default must match the linked build")) return 1;
    if (!smolvla_backend_options_from_env(selected, error)) {
        std::fprintf(stderr, "invalid SmolVLA backend environment: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "graph reuse: CPU%s\n", selected.backend == smolvla_backend::cuda ? " and CUDA" : " (default)");
    error_summary errors;
    outputs cpu_a, cuda_a, cpu, cuda;
    const inputs a(2, 3, 0, false);
    if (!exercise(a, selected, cpu_a, cuda_a, errors)) return 1;
    // Change values, masks and positions while retaining exactly the same keys.
    if (!exercise(inputs(2, 3, 1, false), selected, cpu, cuda, errors)) return 1;
    if (!exercise(inputs(3, 4, 2, true), selected, cpu, cuda, errors)) return 1;
    if (!exercise(a, selected, cpu, cuda, errors) || !same_backend(cpu_a, cpu, "CPU A-B-A")) return 1;
    if (selected.backend == smolvla_backend::cuda && !same_backend(cuda_a, cuda, "CUDA A-B-A")) return 1;
    if (!reject_missing_device()) return 1;

    // Actual expert cross/self-attention and VLM-prefix dimensions. Exercise
    // both probability modes, not just the tiny cache bookkeeping fixtures.
    const size_t shapes[][2] = {{50, 241}, {50, 291}, {241, 241}};
    for (const auto & shape : shapes) {
        for (bool bf16 : {false, true}) {
            if (!exercise(inputs(shape[0], shape[1], 3, bf16), selected, cpu, cuda, errors)) return 1;
        }
    }

    // Nine distinct query-RoPE keys exceed its eight-entry cache; nine distinct
    // attention keys exceed its four-entry cache, with only tiny allocations.
    // Also exercise five-head K RoPE and verify each result against a fresh graph.
    for (size_t n = 4; n <= 12; ++n) {
        if (!exercise(inputs(n, n + 1, int(n), n % 2 != 0), selected, cpu, cuda, errors)) return 1;
    }
    // Recreate A after eviction, then hit it again with changed data and restore
    // A once more. Every recreated/reused graph must still match the baseline.
    for (int pass = 0; pass < 2; ++pass) {
        if (!exercise(a, selected, cpu, cuda, errors) || !same_backend(cpu_a, cpu, "CPU A after eviction")) return 1;
        if (selected.backend == smolvla_backend::cuda && !same_backend(cuda_a, cuda, "CUDA A after eviction")) return 1;
        if (!exercise(inputs(2, 3, 4, false), selected, cpu, cuda, errors)) return 1;
    }
    if (!reject_missing_device()) return 1;
    if (selected.backend == smolvla_backend::cuda) {
        std::fprintf(stderr, "CPU/CUDA overall max_abs: attention_f32=%.9g attention_bf16=%.9g rope=%.9g\n",
                     errors.attention_f32, errors.attention_bf16, errors.rope);
    }
    return 0;
}
