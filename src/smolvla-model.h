#pragma once

#include "llama.h"

struct ggml_tensor;

// Internal bridge: tensors are owned by model and remain valid until it is freed.
// The policy must load all shared weights on its explicitly selected backend.
extern "C" LLAMA_API const struct ggml_tensor * smolvla_model_tensor(
        const struct llama_model * model, const char * name);
