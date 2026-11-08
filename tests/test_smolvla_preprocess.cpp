#include "jetson_pi_smolvla.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
// Frozen scalar oracle from before coordinate reuse. Keep the original expression
// order: a numerically close resize is not sufficient for this optimization.
bool scalar_reference(const uint8_t * rgb, int width, int height, int target_width, int target_height,
                      std::vector<float> & output, std::string & error) {
    output.clear();
    error.clear();
    if (!rgb || width <= 0 || height <= 0 || target_width <= 0 || target_height <= 0 ||
        width > 46000 || height > 46000 || target_width > 46000 || target_height > 46000 ||
        (size_t) target_width > std::numeric_limits<size_t>::max() / (size_t) target_height / 3 ||
        (size_t) width > std::numeric_limits<size_t>::max() / (size_t) height / 3) {
        error = "invalid RGB image dimensions";
        return false;
    }
    const double ratio = std::max((double) width / target_width, (double) height / target_height);
    const int resized_w = (int) (width / ratio);
    const int resized_h = (int) (height / ratio);
    if (resized_w <= 0 || resized_h <= 0) {
        error = "image aspect ratio produces an empty resized image";
        return false;
    }
    const int left = target_width - resized_w;
    const int top = target_height - resized_h;
    output.assign((size_t) target_width * target_height * 3, -1.0f);
    for (int y = 0; y < resized_h; ++y) {
        const float sy = std::max(0.0f, (y + 0.5f) * ((float) height / resized_h) - 0.5f);
        const int y0 = std::min(height - 1, (int) sy);
        const int y1 = std::min(height - 1, y0 + 1);
        const float fy = sy - y0;
        for (int x = 0; x < resized_w; ++x) {
            const float sx = std::max(0.0f, (x + 0.5f) * ((float) width / resized_w) - 0.5f);
            const int x0 = std::min(width - 1, (int) sx);
            const int x1 = std::min(width - 1, x0 + 1);
            const float fx = sx - x0;
            for (int c = 0; c < 3; ++c) {
                const auto pixel = [&](int px, int py) {
                    return (float) rgb[((size_t) py * width + px) * 3 + c] / 255.0f;
                };
                const float a = pixel(x0, y0) * (1.0f - fx) + pixel(x1, y0) * fx;
                const float b = pixel(x0, y1) * (1.0f - fx) + pixel(x1, y1) * fx;
                output[((size_t) (y + top) * target_width + x + left) * 3 + c] =
                    (a * (1.0f - fy) + b * fy) * 2.0f - 1.0f;
            }
        }
    }
    return true;
}

struct shape { int w, h, tw, th; };
size_t cases = 0;
size_t floats_checked = 0;

void check(const uint8_t * rgb, shape s, std::vector<float> & expected, std::vector<float> & actual) {
    expected.assign(9, 7.0f);
    actual.assign(13, 8.0f);
    std::string expected_error = "stale error", actual_error = "stale error";
    const bool a = scalar_reference(rgb, s.w, s.h, s.tw, s.th, expected, expected_error);
    const bool b = smolvla_input::preprocess_rgb(rgb, s.w, s.h, s.tw, s.th, actual, actual_error);
    if (a != b || expected_error != actual_error || expected.size() != actual.size() ||
        (!expected.empty() && std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)))) {
        std::cerr << "preprocess mismatch: " << s.w << 'x' << s.h << " -> " << s.tw << 'x' << s.th << '\n';
        throw std::runtime_error("preprocess status/error/bitwise mismatch");
    }
    ++cases;
    floats_checked += expected.size();
}
} // namespace

int main() {
    try {
        std::mt19937 random_bytes(0x61062026);
        std::vector<float> expected, actual;
        const std::vector<shape> shapes = {
            {256,256,512,512}, {512,512,256,256}, {256,256,256,256},
            {640,480,512,512}, {480,640,512,512}, {319,241,513,511},
            {17,29,31,37}, {31,37,17,29}, {7,3,13,11}, {13,11,7,3},
            {1,1,512,512}, {1,17,32,31}, {17,1,31,32},
            {1,46000,1,257}, {46000,1,257,1}, {46000,1,1,1}, {1,46000,1,1},
            {3,3,1,1}, {2,2,3,3}, {3,2,2,3}, {1,1,1,1}
        };
        std::array<bool,256> seen{};
        for (auto s : shapes) {
            std::vector<uint8_t> rgb((size_t) s.w * s.h * 3);
            for (int pattern = 0; pattern < 5; ++pattern) {
                for (size_t i = 0; i < rgb.size(); ++i) {
                    rgb[i] = pattern == 0 ? (uint8_t) i : pattern == 1 ? (uint8_t) random_bytes() :
                             pattern == 2 ? 0 : pattern == 3 ? 255 : (i % 2 ? 255 : 0);
                    seen[rgb[i]] = true;
                }
                check(rgb.data(), s, expected, actual);
            }
        }
        for (int v = 0; v < 256; ++v) {
            std::vector<uint8_t> rgb(5 * 7 * 3, (uint8_t) v);
            check(rgb.data(), {5,7,13,19}, expected, actual);
        }
        for (int i = 0; i < 1000; ++i) {
            shape s{1 + (int) (random_bytes() % 191), 1 + (int) (random_bytes() % 193),
                    1 + (int) (random_bytes() % 257), 1 + (int) (random_bytes() % 259)};
            std::vector<uint8_t> rgb((size_t) s.w * s.h * 3);
            for (auto & v : rgb) { v = (uint8_t) random_bytes(); }
            check(rgb.data(), s, expected, actual);
        }
        uint8_t pixel[3] = {0,127,255};
        check(nullptr, {1,1,1,1}, expected, actual);
        for (int field = 0; field < 4; ++field) {
            for (int invalid : {std::numeric_limits<int>::min(), -1, 0, 46001, std::numeric_limits<int>::max()}) {
                shape s{1,1,1,1};
                if (field == 0) { s.w = invalid; }
                if (field == 1) { s.h = invalid; }
                if (field == 2) { s.tw = invalid; }
                if (field == 3) { s.th = invalid; }
                check(pixel, s, expected, actual);
                check(pixel, {1,1,3,5}, expected, actual); // immediate failure recovery
            }
        }
        if (!std::all_of(seen.begin(), seen.end(), [](bool v) { return v; })) {
            throw std::runtime_error("incomplete byte coverage");
        }
        std::cout << "PASS preprocess cases=" << cases << " floats=" << floats_checked
                  << " all_byte_values=256 bit_mismatches=0\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
