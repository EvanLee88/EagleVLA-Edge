#pragma once

#include "llama.h"

struct ggml_tensor;

// Internal bridge: tensors are owned by model and remain valid until it is freed.
// The CPU policy must load its model with n_gpu_layers = 0.
extern "C" LLAMA_API const struct ggml_tensor * smolvla_model_tensor(
        const struct llama_model * model, const char * name);
