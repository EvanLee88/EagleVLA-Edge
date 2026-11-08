#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <utility>
#include <vector>

// Pin the oracle's BF16 intermediate as OUTPUT so backend fusion must preserve it.
// All comparisons are byte-exact, including NaN signs/payloads and signed zeros.
static bool run(ggml_backend_t backend, int64_t width, int64_t rows, bool transpose, size_t & comparisons) {
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    if (!ctx) { return false; }
    auto * storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    ggml_set_input(storage);
    auto * input = transpose ? ggml_transpose(ctx.get(), storage) : storage;
    auto * graph = ggml_new_graph(ctx.get());
    ggml_tensor * pinned = nullptr;
    auto round = [&](ggml_tensor * source, bool pin) {
        auto * mid = ggml_cast(ctx.get(), source, GGML_TYPE_BF16);
        if (pin) { ggml_set_output(mid); pinned = mid; }
        auto * out = ggml_cast(ctx.get(), mid, GGML_TYPE_F32);
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
        return out;
    };
    auto * optimized = round(input, false);
    auto * oracle = round(input, true);

    auto * shared = ggml_cast(ctx.get(), input, GGML_TYPE_BF16);
    auto * branch0 = ggml_cast(ctx.get(), shared, GGML_TYPE_F32);
    auto * branch1 = ggml_cast(ctx.get(), shared, GGML_TYPE_F32);
    ggml_set_output(branch0);
    ggml_set_output(branch1);
    ggml_build_forward_expand(graph, branch0);
    ggml_build_forward_expand(graph, branch1);

    auto * viewed = ggml_cast(ctx.get(), input, GGML_TYPE_BF16);
    auto * view_main = ggml_cast(ctx.get(), viewed, GGML_TYPE_F32);
    auto * alias = ggml_view_1d(ctx.get(), viewed, width * rows, 0);
    auto * view_branch = ggml_cast(ctx.get(), alias, GGML_TYPE_F32);
    ggml_set_output(view_main);
    ggml_set_output(view_branch);
    ggml_build_forward_expand(graph, view_main);
    ggml_build_forward_expand(graph, view_branch);

    // Two separate compute graphs make a guaranteed unfused oracle, independent
    // of whether the candidate correctly honors the main graph's OUTPUT flag.
    auto * ref_mid = ggml_cast(ctx.get(), input, GGML_TYPE_BF16);
    auto * ref_input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, input->ne[0], input->ne[1]);
    ggml_set_input(ref_input);
    auto * ref_out = ggml_cast(ctx.get(), ref_input, GGML_TYPE_F32);
    auto * ref_graph0 = ggml_new_graph(ctx.get());
    auto * ref_graph1 = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(ref_graph0, ref_mid);
    ggml_build_forward_expand(ref_graph1, ref_out);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buffer) { return false; }
    const size_t count = size_t(width) * rows;
    std::vector<float> input_data(count), expected(count), actual(count);
    std::vector<uint16_t> mid_bits(count), pinned_bits(count);
    std::mt19937 rng(0x11062026);
    const uint32_t low_bits[] = {0, 1, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff};
    const uint32_t specials[] = {0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001,
        0x7f800001, 0xffc12345, 0xff812345, 0x7f7fffff, 0xff7fffff, 1, 0x80000001};
    for (int pattern : {0, 1, 2, 0}) {
        for (size_t i = 0; i < count; ++i) {
            uint32_t bits = pattern == 0 ? rng() : pattern == 1 ?
                (uint32_t((i / 7) & 0xffff) << 16) | low_bits[i % 7] : specials[i % 12];
            std::memcpy(&input_data[i], &bits, sizeof(bits));
        }
        ggml_backend_tensor_set(storage, input_data.data(), 0, count * sizeof(float));
        if (ggml_backend_graph_compute(backend, ref_graph0) != GGML_STATUS_SUCCESS) { return false; }
        ggml_backend_tensor_get(ref_mid, mid_bits.data(), 0, count * sizeof(uint16_t));
        ggml_backend_tensor_set(ref_input, mid_bits.data(), 0, count * sizeof(uint16_t));
        if (ggml_backend_graph_compute(backend, ref_graph1) != GGML_STATUS_SUCCESS) { return false; }
        ggml_backend_tensor_get(ref_out, expected.data(), 0, count * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return false; }
        ggml_backend_tensor_get(pinned, pinned_bits.data(), 0, count * sizeof(uint16_t));
        if (std::memcmp(mid_bits.data(), pinned_bits.data(), count * sizeof(uint16_t))) {
            std::fputs("BF16 graph OUTPUT was not preserved\n", stderr);
            return false;
        }
        for (auto * out : {optimized, oracle, branch0, branch1, view_main, view_branch}) {
            ggml_backend_tensor_get(out, actual.data(), 0, count * sizeof(float));
            if (std::memcmp(expected.data(), actual.data(), count * sizeof(float))) {
                std::fprintf(stderr, "BF16 round mismatch shape=%lldx%lld transpose=%d pattern=%d\n",
                    (long long) width, (long long) rows, int(transpose), pattern);
                return false;
            }
            comparisons += count;
        }
    }
    return true;
}

// Explicit backing-memory overlap without view metadata on the destination:
// old two-launch cast is safe for all offsets; fused direct copy is only safe
// for exact in-place or disjoint ranges.
static bool run_alias(ggml_backend_t backend, int offset, size_t & comparisons) {
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    if (!ctx) { return false; }
    const int64_t n = 4096;
    auto * storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2 * n + 2);
    ggml_set_input(storage);
    const size_t start = offset < 0 ? 1 : 0;
    auto * input = ggml_view_1d(ctx.get(), storage, n, start * sizeof(float));
    auto * mid = ggml_cast(ctx.get(), input, GGML_TYPE_BF16);
    auto * out = ggml_cast(ctx.get(), mid, GGML_TYPE_F32);
    ggml_set_output(out);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    auto * ref_mid = ggml_cast(ctx.get(), input, GGML_TYPE_BF16);
    auto * ref_input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_BF16, n);
    auto * ref_out = ggml_cast(ctx.get(), ref_input, GGML_TYPE_F32);
    auto * graph0 = ggml_new_graph(ctx.get());
    auto * graph1 = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph0, ref_mid);
    ggml_build_forward_expand(graph1, ref_out);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buffer) { return false; }
    out->data = static_cast<char *>(storage->data) + (int64_t(start) + offset) * sizeof(float);
    std::vector<float> data(2 * n + 2), expected(n), actual(n);
    std::vector<uint16_t> rounded(n);
    std::mt19937 rng(0x11062026);
    for (int repeat = 0; repeat < 8; ++repeat) {
        for (auto & value : data) {
            uint32_t bits = rng();
            std::memcpy(&value, &bits, sizeof(bits));
        }
        ggml_backend_tensor_set(storage, data.data(), 0, data.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph0) != GGML_STATUS_SUCCESS) { return false; }
        ggml_backend_tensor_get(ref_mid, rounded.data(), 0, n * sizeof(uint16_t));
        ggml_backend_tensor_set(ref_input, rounded.data(), 0, n * sizeof(uint16_t));
        if (ggml_backend_graph_compute(backend, graph1) != GGML_STATUS_SUCCESS) { return false; }
        ggml_backend_tensor_get(ref_out, expected.data(), 0, n * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return false; }
        ggml_backend_tensor_get(out, actual.data(), 0, n * sizeof(float));
        if (std::memcmp(expected.data(), actual.data(), n * sizeof(float))) {
            std::fprintf(stderr, "BF16 overlap mismatch offset=%d repeat=%d\n", offset, repeat);
            return false;
        }
        comparisons += n;
    }
    return true;
}

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) { std::puts("SKIP: CUDA unavailable"); return 77; }
    using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    backend_ptr backend(ggml_backend_cuda_init(0), ggml_backend_free);
    if (!backend) { return 1; }
    size_t comparisons = 0;
    for (int64_t n : {int64_t(1), int64_t(31), int64_t(255), int64_t(256), int64_t(257), int64_t(65536 * 7)}) {
        if (!run(backend.get(), n, 1, false, comparisons)) { return 1; }
    }
    for (auto shape : {std::pair<int64_t, int64_t>{720, 50}, {2048, 50}, {960, 241}}) {
        if (!run(backend.get(), shape.first, shape.second, false, comparisons) ||
            !run(backend.get(), shape.first, shape.second, true, comparisons)) { return 1; }
    }
    for (int offset : {-1, 0, 1, 4096}) {
        if (!run_alias(backend.get(), offset, comparisons)) { return 1; }
    }
    std::printf("PASS BF16 round byte-exact values=%zu (random bits, all BF16 neighbors, special values, branches, views, transpose, repeated updates)\n", comparisons);
    return 0;
}
