#include "jetson_pi_smolvla.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > 1e-6f) return false;
    return true;
}

int main() {
    std::string error;
    for (int pass = 0; pass < 3; ++pass) {
        const size_t nq = pass == 1 ? 3 : 2, nk = pass == 1 ? 4 : 3;
        std::vector<float> q(nq * 15 * 64), k(nk * 5 * 64), v(k.size());
        for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(float(i + pass) * 0.03f);
        for (size_t i = 0; i < k.size(); ++i) { k[i] = std::cos(float(i + pass) * 0.02f); v[i] = std::sin(float(i + pass) * 0.01f); }
        std::vector<uint8_t> mask(nq * nk, 1);
        if (pass == 2) mask[0] = 0;
        std::vector<float> reused, fresh, rotated, rotated_fresh;
        std::vector<int32_t> pos(nq);
        for (size_t i = 0; i < nq; ++i) pos[i] = int32_t(i + pass);
        const bool round = pass != 1;
        if (!smolvla::attention(q.data(), k.data(), v.data(), nq, nk, mask.data(), 1, reused, error, round) ||
            !smolvla::expert_rope(q.data(), nq, 15, pos.data(), 1, rotated, error)) {
            std::fprintf(stderr, "cached graph failed: %s\n", error.c_str()); return 1;
        }
        bool ok = false;
        std::thread worker([&] {
            std::string thread_error;
            ok = smolvla::attention(q.data(), k.data(), v.data(), nq, nk, mask.data(), 1, fresh, thread_error, round) &&
                 smolvla::expert_rope(q.data(), nq, 15, pos.data(), 1, rotated_fresh, thread_error);
        });
        worker.join();
        if (!ok || !same(reused, fresh) || !same(rotated, rotated_fresh)) {
            std::fprintf(stderr, "cached graph differs from fresh graph on pass %d\n", pass); return 1;
        }
    }
    return 0;
}
