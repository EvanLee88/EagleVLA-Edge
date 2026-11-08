#pragma once

// SmolVLA tokenizer and policy interfaces.

#include "llama.h"
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
struct smolvla_tokenizer;
// Load tokenizer metadata without loading model weights. Returns NULL on error.
LLAMA_API struct smolvla_tokenizer * smolvla_tokenizer_open(const char * gguf_path);
LLAMA_API void smolvla_tokenizer_free(struct smolvla_tokenizer * tokenizer);
// Append a newline when missing; tokenize, truncate and right-pad to 48 tokens.
// `valid` contains 1 for non-padding tokens. Returns 0 on success.
LLAMA_API int32_t smolvla_tokenizer_encode(const struct smolvla_tokenizer * tokenizer,
                                           const char * text, size_t length,
                                           int32_t tokens[48], uint8_t valid[48]);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

// input

#include <cstdint>
#include <string>
#include <vector>

enum class smolvla_backend { cpu, cuda };
// Defined by the linked library's build: CUDA when GGML_CUDA=ON, otherwise CPU.
// Keep this out of header conditionals so external consumers share the same default.
LLAMA_API smolvla_backend smolvla_default_backend();
struct smolvla_backend_options {
    smolvla_backend backend = smolvla_default_backend();
    std::string device = "CUDA0";
};

// Parse optional runtime overrides; unset variables preserve the build default.
// Library open() defaults follow the build, without reading environment variables.
LLAMA_API bool smolvla_backend_options_from_env(smolvla_backend_options & options, std::string & error);

namespace smolvla_input {

// Resize RGB bytes with bilinear sampling, pad on the top/left, and map to
// [-1, 1]. Output is interleaved RGB float32, ready for the vision encoder.
bool preprocess_rgb(const uint8_t * rgb, int width, int height,
                    int target_width, int target_height,
                    std::vector<float> & output, std::string & error);

} // namespace smolvla_input

namespace smolvla_input {
// Builds the prefix attention mask (query-major) and position IDs for ordered
// image tokens, language tokens, and the final state token. Mask values are 0/1.
bool build_prefix_attention(const std::vector<uint8_t> & image_valid,
                            int tokens_per_image,
                            const std::vector<uint8_t> & language_valid,
                            std::vector<uint8_t> & padding,
                            std::vector<uint8_t> & attention,
                            std::vector<int32_t> & positions,
                            std::string & error);
}

namespace smolvla_input {
// Query-major suffix-to-prefix-plus-suffix mask for a self-attention layer,
// and suffix-to-prefix mask for a cross-attention layer. `prefix_valid` is the
// prefill padding mask; positions are the absolute self-attention positions.
bool build_suffix_attention(const std::vector<uint8_t> & prefix_valid, size_t action_steps,
                            std::vector<uint8_t> & self_attention,
                            std::vector<uint8_t> & cross_attention,
                            std::vector<int32_t> & positions, std::string & error);
}


// flow

#include <functional>
#include <string>
#include <vector>

namespace smolvla {
// Integrate a velocity field from t=1 to t=0 with a fixed Euler schedule.
// The callback receives the current state and time; it must return a finite
// velocity of the same size. A failure leaves the caller's output empty.
using velocity_function = std::function<bool(const std::vector<float> &, float, std::vector<float> &, std::string &)>;
bool euler_integrate(const std::vector<float> & noise, int num_steps, const velocity_function & velocity,
                     std::vector<float> & output, std::string & error);
}


// action input

#include <memory>
#include <string>
#include <vector>

// Action input and output projections; attention expert execution is separate.
class smolvla_action_input {
public:
    static std::unique_ptr<smolvla_action_input> open(const char * path, int threads, std::string & error, const smolvla_backend_options & backend = {});
    // The caller keeps the selected-backend llama model alive for the lifetime of this object.
    static std::unique_ptr<smolvla_action_input> open(const llama_model * model, int threads, std::string & error, const smolvla_backend_options & backend = {});
    ~smolvla_action_input();
    smolvla_action_input(const smolvla_action_input &) = delete;
    smolvla_action_input & operator=(const smolvla_action_input &) = delete;
    // 50 x 32 normalized noisy actions, t in [0,1], output 50 x 720.
    bool encode(const float * noisy_actions, float time, std::vector<float> & output, std::string & error);
    // Project an already normalized 32-element state into a 960-element prefix.
    bool project_state(const float * state, std::vector<float> & embedding, std::string & error);
    // Apply BF16 expert final RMS normalization to 50 x 720 hidden values.
    bool normalize_expert(const float * hidden, std::vector<float> & output, std::string & error);
    // Project 50 x 720 final expert outputs into 50 x 32 velocity values.
    bool project_velocity(const float * expert_output, std::vector<float> & velocity, std::string & error);
private:
    friend class smolvla_expert;
    smolvla_action_input();
    struct impl;
    std::unique_ptr<impl> state;
};


// rope
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace smolvla {
// Apply LeRobot's split-half RoPE to [token, head, 64] float32 values.
// This is for the unpermuted expert Q/K weights; the converted VLM Q/K weights
// use the existing llama.cpp interleaved layout instead.
bool expert_rope(const float * input, size_t tokens, size_t heads,
                 const int32_t * positions, int threads,
                 std::vector<float> & output, std::string & error, const smolvla_backend_options & backend = {});
}


// attention
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace smolvla {
// Inputs: query [query,15,64], key/value [key,5,64], mask [query,key].
// Output: [query,15,64]. Heads follow LeRobot's grouped-query expansion.
bool attention(const float * query, const float * key, const float * value,
               size_t query_count, size_t key_count, const uint8_t * mask,
               int threads, std::vector<float> & output, std::string & error,
               bool bf16_probabilities = false, const smolvla_backend_options & backend = {});
}


// expert mlp
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Single SmolVLA expert MLP, for incremental runtime validation.
class smolvla_expert_mlp {
public:
    static std::unique_ptr<smolvla_expert_mlp> open(const char * model, int layer, int threads, std::string & error, const smolvla_backend_options & backend = {});
    // The caller keeps the selected-backend llama model alive for the lifetime of this object.
    static std::unique_ptr<smolvla_expert_mlp> open(const llama_model * model, int layer, int threads, std::string & error, const smolvla_backend_options & backend = {});
    ~smolvla_expert_mlp();
    smolvla_expert_mlp(const smolvla_expert_mlp &) = delete;
    smolvla_expert_mlp & operator=(const smolvla_expert_mlp &) = delete;
    // Q from [50,720] hidden states; self-attention K/V use hidden states,
    // cross-attention K/V use [tokens,320] prefix cache values.
    bool project_qkv(const float * hidden, const float * cache_k, const float * cache_v,
                     size_t cache_tokens, std::vector<float> & q, std::vector<float> & k,
                     std::vector<float> & v, std::string & error);
    // Complete one expert layer using post-RoPE prefix K/V in [token,5,64].
    bool forward(const float * hidden, const float * prefix_k, const float * prefix_v,
                 const std::vector<uint8_t> & prefix_valid, std::vector<float> & output,
                 std::string & error);
    // Combine attention output [50,960] and hidden state [50,720], apply
    // output projection, RMSNorm, MLP and residuals.
    bool finish_layer(const float * hidden, const float * attention, std::vector<float> & output, std::string & error);
    // Input/output: 50 tokens of 720 elements. Input is already normalized.
    bool run(const float * input, std::vector<float> & output, std::string & error);
private:
    friend class smolvla_expert;
    struct sample_workspace;
    bool prepare_cross_kv(const float * cache_k, const float * cache_v, size_t tokens,
                          sample_workspace & workspace, std::string & error);
    bool project_qkv_impl(const float * hidden, const float * cache_k, const float * cache_v,
                          size_t cache_tokens, std::vector<float> & q, std::vector<float> & k,
                          std::vector<float> & v, std::string & error, sample_workspace * workspace);
    bool forward_resident(const float * hidden, const float * prefix_k, const float * prefix_v,
                          size_t prefix_tokens, std::vector<float> & output,
                          std::string & error, sample_workspace & workspace);
    bool forward_impl(const float * hidden, const float * prefix_k, const float * prefix_v,
                      const std::vector<uint8_t> & prefix_valid, std::vector<float> & output,
                      std::string & error, sample_workspace * workspace);
    smolvla_expert_mlp();
    struct impl;
    std::unique_ptr<impl> state;
};


// expert
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Action expert for the smolvla_base architecture.
// Cache vectors contain one [prefix_tokens, 5, 64] tensor per layer.
class smolvla_expert {
public:
    static std::unique_ptr<smolvla_expert> open(const char * path, int threads, std::string & error, const smolvla_backend_options & backend = {});
    // The caller keeps the selected-backend llama model alive for the lifetime of this object.
    static std::unique_ptr<smolvla_expert> open(const llama_model * model, int threads, std::string & error, const smolvla_backend_options & backend = {});
    ~smolvla_expert();
    smolvla_expert(const smolvla_expert &) = delete;
    smolvla_expert & operator=(const smolvla_expert &) = delete;
    // Share the expert's projection weights when assembling the state prefix.
    bool project_state(const float * state, std::vector<float> & embedding, std::string & error);
    bool forward(const float * suffix, const std::vector<std::vector<float>> & keys,
                 const std::vector<std::vector<float>> & values, const std::vector<uint8_t> & prefix_mask,
                 std::vector<float> & normalized_output, std::string & error);
    // Generate a normalized 50 x 32 action chunk from fixed noise and prefix KV.
    bool sample(const std::vector<float> & noise, const std::vector<std::vector<float>> & keys,
                const std::vector<std::vector<float>> & values, const std::vector<uint8_t> & prefix_mask,
                std::vector<float> & actions, std::string & error);
private:
    smolvla_expert();
    struct impl;
    std::unique_ptr<impl> state;
};


// prefix
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Implementation of the 16-layer SmolVLA VLM prefix.
class smolvla_prefix {
public:
    static std::unique_ptr<smolvla_prefix> open(const char * path, int threads, std::string & error, const smolvla_backend_options & backend = {});
    // The caller keeps the selected-backend llama model alive for the lifetime of this object.
    static std::unique_ptr<smolvla_prefix> open(const llama_model * model, int threads, std::string & error, const smolvla_backend_options & backend = {});
    ~smolvla_prefix();
    smolvla_prefix(const smolvla_prefix &) = delete;
    smolvla_prefix & operator=(const smolvla_prefix &) = delete;
    // Embed token IDs [0,49152...] as 960-wide language tokens. Returned
    // embeddings include SmolVLA's sqrt(960) scaling.
    bool embed_tokens(const std::vector<int32_t> & tokens, std::vector<float> & output, std::string & error);
    // Assemble three 64-token image embeddings, 48 language IDs and a
    // projected state. Image embeddings must be unscaled.
    bool assemble(const std::vector<std::vector<float>> & images, const std::vector<int32_t> & tokens,
                  const std::vector<uint8_t> & language_valid, const std::vector<float> & state_embedding,
                  std::vector<float> & prefix, std::vector<uint8_t> & padding,
                  std::vector<uint8_t> & attention, std::vector<int32_t> & positions, std::string & error);
    // prefix [tokens,960], mask [tokens,tokens], positions [tokens]; caches
    // are returned in [layer][tokens,5,64] layout.
    bool forward(const std::vector<float> & prefix, const std::vector<uint8_t> & mask,
                 const std::vector<int32_t> & positions, std::vector<std::vector<float>> & keys,
                 std::vector<std::vector<float>> & values, std::string & error);
private:
    smolvla_prefix();
    struct impl;
    std::unique_ptr<impl> state;
};


// vision

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// SmolVLA image embeddings only; this does not run the action policy.
class smolvla_vision {
public:
    static std::unique_ptr<smolvla_vision> open(const char * mmproj_path, int threads, std::string & error, const smolvla_backend_options & backend = {});
    ~smolvla_vision();
    smolvla_vision(const smolvla_vision &) = delete;
    smolvla_vision & operator=(const smolvla_vision &) = delete;

    // `rgb` is interleaved, with width*height*3 bytes. Results are laid out as
    // [token_count, embedding_width]. A handle must not be used concurrently.
    bool encode(const uint8_t * rgb, int width, int height, std::vector<float> & embeddings,
                int & token_count, int & embedding_width, std::string & error);

private:
    smolvla_vision();
    struct impl;
    std::unique_ptr<impl> state;
};


// policy
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct smolvla_rgb_image { const uint8_t * data; int width; int height; };
// Completed stage wall times, including host work and boundary transfers, in ms.
struct smolvla_policy_timings {
    double vit_ms = 0;              // All three image encodes, including preprocessing/projector.
    double prefix_prepare_ms = 0;   // State projection and prefix assembly.
    double llm_prefill_ms = 0;      // Prefix KV computation, not autoregressive decoding.
    double action_expert_ms = 0;    // All ten sampling steps, projections and integration.
    double total_ms = 0;            // Full infer(), excluding model loading.
};
// Inference for the supplied smolvla_base checkpoint. The caller provides
// normalized state and deterministic initial noise; actions are normalized.
// Instances contain mutable graph caches and must not be used concurrently.
class smolvla_policy {
public:
    static std::unique_ptr<smolvla_policy> open(const char * model, const char * mmproj, int threads, std::string & error, const smolvla_backend_options & backend = {});
    ~smolvla_policy();
    smolvla_policy(const smolvla_policy &) = delete;
    smolvla_policy & operator=(const smolvla_policy &) = delete;
    // Optional timings are cleared on entry and populated only on success.
    bool infer(const std::vector<smolvla_rgb_image> & images, const std::string & instruction,
               const std::vector<float> & normalized_state, const std::vector<float> & noise,
               std::vector<float> & normalized_actions, std::string & error,
               smolvla_policy_timings * timings = nullptr);
private:
    smolvla_policy();
    struct impl;
    std::unique_ptr<impl> state;
};

#endif // __cplusplus
