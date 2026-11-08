#include "jetson_pi_smolvla.h"

#include <cstdio>
#include <string>

// Expected values come from CMake, not from consumer-side backend macros.
// This target links only llama and must share the library's default selection.
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const auto expected_default = std::string(argv[1]) == "cuda" ? smolvla_backend::cuda : smolvla_backend::cpu;
    smolvla_backend_options options;
    if (smolvla_default_backend() != expected_default || options.backend != expected_default || options.device != "CUDA0") {
        std::fprintf(stderr, "library default does not match GGML_CUDA\n");
        return 1;
    }
    // Parsing must reset previous values, and explicit cpu must override CUDA builds.
    options.backend = expected_default == smolvla_backend::cuda ? smolvla_backend::cpu : smolvla_backend::cuda;
    options.device = "stale";
    std::string error;
    const bool ok = smolvla_backend_options_from_env(options, error);
    const std::string expected_parse = argv[2];
    if (expected_parse == "error") {
        if (ok || error.empty()) return 1;
    } else {
        const auto expected = expected_parse == "cuda" ? smolvla_backend::cuda : smolvla_backend::cpu;
        if (!ok || !error.empty() || options.backend != expected || options.device != "CUDA0") {
            std::fprintf(stderr, "unexpected runtime selection: %s\n", error.c_str());
            return 1;
        }
    }
    std::puts("PASS: build default and optional runtime override");
    return 0;
}
