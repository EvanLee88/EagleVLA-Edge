#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

// A zero F32 mask forces the generic softmax kernel. For these finite inputs,
// it has exactly the same arithmetic as no mask, providing a live old-kernel
// oracle without duplicating a CUDA kernel in the test executable.
int main() {
    using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    if (ggml_backend_cuda_get_device_count() == 0) { std::puts("SKIP: CUDA unavailable"); return 77; }
    backend_ptr backend(ggml_backend_cuda_init(0), ggml_backend_free);
    if (!backend) { std::puts("SKIP: CUDA unavailable"); return 77; }
    context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    if (!ctx) { return 1; }
    const int columns = 1024, rows = 1024, heads = 12;
    auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, columns, rows, heads);
    auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, columns, rows);
    ggml_set_input(input);
    ggml_set_input(mask);
    std::vector<ggml_tensor *> optimized, fallback;
    auto * graph = ggml_new_graph(ctx.get());
    for (float scale : {0.125f, 1.0f}) {
        optimized.push_back(ggml_soft_max_ext(ctx.get(), input, nullptr, scale, 0.0f));
        fallback.push_back(ggml_soft_max_ext(ctx.get(), input, mask, scale, 0.0f));
        ggml_set_output(optimized.back());
        ggml_set_output(fallback.back());
        ggml_build_forward_expand(graph, optimized.back());
        ggml_build_forward_expand(graph, fallback.back());
    }
    // Destroy the backend allocation before the metadata context and backend.
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer) { return 1; }
    const size_t count = (size_t) columns * rows * heads;
    std::vector<float> data(count), zeros((size_t) columns * rows, 0.0f), a(count), b(count);
    ggml_backend_tensor_set(mask, zeros.data(), 0, zeros.size() * sizeof(float));
    std::mt19937 rng(0x61062026);
    int checks = 0;
    for (int pattern : {0, 1, 2, 3, 0}) {
        for (size_t i = 0; i < count; ++i) {
            if (pattern == 0) { data[i] = (int(rng() % 8193) - 4096) / 128.0f; }
            if (pattern == 1) { data[i] = 0.0f; }
            if (pattern == 2) { data[i] = i % 2 ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max(); }
            if (pattern == 3) { data[i] = i % 2 ? -0.0f : 0.0f; }
        }
        ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) { return 1; }
        for (size_t mode = 0; mode < optimized.size(); ++mode) {
            ggml_backend_tensor_get(optimized[mode], a.data(), 0, a.size() * sizeof(float));
            ggml_backend_tensor_get(fallback[mode], b.data(), 0, b.size() * sizeof(float));
            if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float))) {
                std::fprintf(stderr, "softmax bitwise mismatch pattern=%d scale_index=%zu\n", pattern, mode);
                return 1;
            }
            for (float value : a) {
                if (!std::isfinite(value) || value < 0.0f) { return 1; }
            }
            if (pattern == 1 || pattern == 3) {
                for (float value : a) {
                    if (value != 1.0f / columns) { return 1; }
                }
            }
            ++checks;
        }
    }
    std::printf("PASS vision softmax bitwise fallback comparisons=%d values_per_comparison=%zu\n", checks, count);
    return 0;
}
