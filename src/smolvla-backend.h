#pragma once

#include "jetson_pi_smolvla.h"
#include "ggml-backend.h"

#include <cstring>
#include <mutex>

// Resolve an explicit device, never silently substitute CPU for a CUDA request.
inline ggml_backend_dev_t smolvla_backend_device(const smolvla_backend_options & options, std::string & error) {
    static std::once_flag loaded;
    std::call_once(loaded, [] { ggml_backend_load_all(); });
    if (options.backend == smolvla_backend::cpu) {
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!device) error = "SmolVLA CPU backend is unavailable";
        return device;
    }
    if (options.backend != smolvla_backend::cuda) {
        error = "invalid SmolVLA backend";
        return nullptr;
    }
    auto * device = ggml_backend_dev_by_name(options.device.c_str());
    if (!device || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") != 0) {
        error = "SmolVLA CUDA device is unavailable: " + options.device;
        return nullptr;
    }
    return device;
}
