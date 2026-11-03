#include "jetson_pi_smolvla.h"

// These sections are compiled into separate targets to preserve their dependency boundaries.

#if defined(SMOLVLA_BUILD_TOKENIZER)

// tokenizer
#include "llama-model-loader.h"
#include "llama-model.h"
#include "smolvla-model.h"
#include "llama-vocab.h"
#include "llama-arch.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

const ggml_tensor * smolvla_model_tensor(const llama_model * model, const char * name) {
    if (!model || model->arch != LLM_ARCH_SMOLVLA || !name) return nullptr;
    return model->get_tensor(name);
}

struct smolvla_tokenizer { llama_vocab vocab; };
smolvla_tokenizer * smolvla_tokenizer_open(const char * path) {
    if (!path || !*path) return nullptr;
    try {
        std::vector<std::string> splits;
        llama_model_loader loader(nullptr,nullptr,nullptr,path,splits,nullptr,false,false,false,true,nullptr,nullptr);
        if (loader.get_arch_name() != "smolvla") return nullptr;
        auto * result = new smolvla_tokenizer;
        try { result->vocab.load(loader, LLM_KV(LLM_ARCH_UNKNOWN)); }
        catch (...) { delete result; throw; }
        return result;
    } catch (const std::exception &) { return nullptr; }
}
void smolvla_tokenizer_free(smolvla_tokenizer * tokenizer) { delete tokenizer; }
int32_t smolvla_tokenizer_encode(const smolvla_tokenizer * tokenizer, const char * text, size_t length,
                                  int32_t tokens[48], uint8_t valid[48]) {
    if (!tokenizer || !text || !tokens || !valid || length > INT32_MAX - 1) return -1;
    try {
        std::string instruction(text,length);
        if (instruction.empty() || instruction.back() != '\n') instruction.push_back('\n');
        auto ids = tokenizer->vocab.tokenize(instruction,true,true);
        const auto pad = tokenizer->vocab.token_pad();
        if (pad < 0) return -1;
        const size_t n = std::min<size_t>(48,ids.size());
        const size_t start = ids.size() - n; // SmolVLM2 tokenizer truncates on the left.
        for (size_t i=0;i<48;++i) { tokens[i] = i<n ? ids[start+i] : pad; valid[i] = i<n ? 1 : 0; }
        return 0;
    } catch (const std::exception &) { return -1; }
}

#endif // SMOLVLA_BUILD_TOKENIZER

#if defined(SMOLVLA_BUILD_INPUT)

// input

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace smolvla_input {

bool preprocess_rgb(const uint8_t * rgb, int width, int height,
                    int target_width, int target_height,
                    std::vector<float> & output, std::string & error) {
    output.clear();
    error.clear();
    // Match the image size constraints used by mtmd and prevent allocation overflow.
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

} // namespace smolvla_input

namespace smolvla_input {
bool build_prefix_attention(const std::vector<uint8_t> & image_valid, int tokens_per_image,
                            const std::vector<uint8_t> & language_valid, std::vector<uint8_t> & padding,
                            std::vector<uint8_t> & attention, std::vector<int32_t> & positions,
                            std::string & error) {
    padding.clear(); attention.clear(); positions.clear(); error.clear();
    // Prevent unreasonable allocations and integer overflow from untrusted inputs.
    if (tokens_per_image <= 0 || image_valid.size() > 64 || language_valid.size() > 8192 ||
        tokens_per_image > 8192 || image_valid.size() * (size_t) tokens_per_image + language_valid.size() + 1 > 8192) {
        error = "invalid prefix dimensions"; return false;
    }
    for (uint8_t valid : image_valid) {
        if (valid > 1) { error = "invalid image mask"; return false; }
    }
    for (uint8_t valid : language_valid) {
        if (valid > 1) { error = "invalid language mask"; return false; }
    }
    for (uint8_t valid : image_valid) padding.insert(padding.end(), tokens_per_image, valid);
    padding.insert(padding.end(), language_valid.begin(), language_valid.end());
    padding.push_back(1); // state
    const size_t n = padding.size();
    attention.resize(n * n);
    positions.resize(n);
    int32_t position = -1;
    for (size_t query = 0; query < n; ++query) {
        position += padding[query];
        positions[query] = position;
        for (size_t key = 0; key < n; ++key) {
            // All image/language tokens share group zero. State is group one.
            const bool visible = key != n - 1 || query == n - 1;
            attention[query * n + key] = padding[query] && padding[key] && visible;
        }
    }
    return true;
}
}

namespace smolvla_input {
bool build_suffix_attention(const std::vector<uint8_t> & prefix_valid, size_t action_steps,
                            std::vector<uint8_t> & self_attention, std::vector<uint8_t> & cross_attention,
                            std::vector<int32_t> & positions, std::string & error) {
    self_attention.clear(); cross_attention.clear(); positions.clear(); error.clear();
    const size_t prefix = prefix_valid.size();
    if (prefix == 0 || prefix > 8192 || action_steps == 0 || action_steps > 8192 ||
        prefix + action_steps > 8192) { error = "invalid suffix dimensions"; return false; }
    int32_t valid_count = 0;
    for (uint8_t valid : prefix_valid) {
        if (valid > 1) { error = "invalid prefix mask"; return false; }
        valid_count += valid;
    }
    self_attention.resize(action_steps * (prefix + action_steps));
    cross_attention.resize(action_steps * prefix);
    positions.resize(action_steps);
    for (size_t query = 0; query < action_steps; ++query) {
        positions[query] = valid_count + static_cast<int32_t>(query);
        for (size_t key = 0; key < prefix; ++key) {
            cross_attention[query * prefix + key] = prefix_valid[key];
            self_attention[query * (prefix + action_steps) + key] = prefix_valid[key];
        }
        for (size_t key = 0; key < action_steps; ++key) {
            self_attention[query * (prefix + action_steps) + prefix + key] = key <= query;
        }
    }
    return true;
}
}


// flow

#include <cmath>
#include <exception>
#include <utility>

namespace smolvla {
bool euler_integrate(const std::vector<float> & noise, int num_steps, const velocity_function & velocity,
                     std::vector<float> & output, std::string & error) {
    // Make a copy first so a caller can pass the same vector as input and output.
    std::vector<float> current = noise;
    output.clear(); error.clear();
    if (current.empty() || num_steps <= 0 || !velocity) { error = "invalid flow integration input"; return false; }
    for (float value : current) {
        if (!std::isfinite(value)) { error = "non-finite initial noise"; return false; }
    }
    const float dt = -1.0f / num_steps;
    try {
        for (int step = 0; step < num_steps; ++step) {
            const float time = static_cast<float>(1.0 - static_cast<double>(step) / num_steps);
            std::vector<float> v;
            if (!velocity(current, time, v, error)) {
                if (error.empty()) error = "velocity computation failed";
                return false;
            }
            if (v.size() != current.size()) { error = "velocity shape mismatch"; return false; }
            for (size_t i = 0; i < current.size(); ++i) {
                if (!std::isfinite(v[i])) { error = "non-finite velocity"; return false; }
                current[i] += dt * v[i];
                if (!std::isfinite(current[i])) { error = "non-finite integrated action"; return false; }
            }
        }
    } catch (const std::exception & e) { error = e.what(); return false; }
    output = std::move(current);
    return true;
}
}

#endif // SMOLVLA_BUILD_INPUT

#if defined(SMOLVLA_BUILD_VISION)

// vision
#include "clip-impl.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <thread>

struct smolvla_vision::impl {
    clip_ctx * vision = nullptr;
    int threads = 1;
    ~impl() { if (vision) clip_free(vision); }
};

smolvla_vision::smolvla_vision() : state(new impl) {}
smolvla_vision::~smolvla_vision() = default;

std::unique_ptr<smolvla_vision> smolvla_vision::open(const char * path, int threads, std::string & error) {
    error.clear();
    if (!path || !*path || threads < 0) {
        error = "invalid SmolVLA vision configuration";
        return nullptr;
    }
    try {
        auto handle = std::unique_ptr<smolvla_vision>(new smolvla_vision);
        handle->state->threads = threads ? threads : std::max(1u, std::thread::hardware_concurrency());
        gguf_init_params metadata_params{true, nullptr};
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file(path, metadata_params), gguf_free);
        if (!metadata) {
            error = "failed to read vision GGUF metadata";
            return nullptr;
        }
        const int64_t size_key = gguf_find_key(metadata.get(), "clip.vision.image_size");
        if (size_key < 0 || gguf_get_kv_type(metadata.get(), size_key) != GGUF_TYPE_UINT32 ||
            gguf_get_val_u32(metadata.get(), size_key) != 512) {
            error = "SmolVLA vision encoder must use 512x512 images";
            return nullptr;
        }
        ggml_backend_load_all();
        clip_context_params params{};
        params.use_gpu = false;
        params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
        params.warmup = false;
        auto result = clip_init(path, params);
        handle->state->vision = result.ctx_v;
        if (result.ctx_a) clip_free(result.ctx_a);
        if (!handle->state->vision || clip_get_projector_type(handle->state->vision) != PROJECTOR_TYPE_IDEFICS3) {
            error = "failed to load an Idefics3 vision projector";
            return nullptr;
        }
        return handle;
    } catch (const std::exception & e) {
        error = e.what();
        return nullptr;
    }
}

bool smolvla_vision::encode(const uint8_t * rgb, int width, int height, std::vector<float> & embeddings,
                            int & tokens, int & dimension, std::string & error) {
    embeddings.clear();
    tokens = dimension = 0;
    error.clear();
    try {
        std::vector<float> pixels;
        // The currently supported SmolVLA base checkpoint uses a 512x512 encoder.
        if (!smolvla_input::preprocess_rgb(rgb, width, height, 512, 512, pixels, error)) return false;
        clip_image_f32 image;
        image.set_size({512, 512}, false, false);
        image.cpy_buf(pixels);
        const int n_tokens = clip_n_output_tokens(state->vision, &image);
        const int n_embd = clip_n_mmproj_embd(state->vision);
        if (n_tokens <= 0 || n_embd <= 0 || (size_t) n_tokens > std::numeric_limits<size_t>::max() / (size_t) n_embd) {
            error = "invalid vision output dimensions";
            return false;
        }
        embeddings.resize((size_t) n_tokens * n_embd);
        if (!clip_image_encode(state->vision, state->threads, &image, embeddings)) {
            embeddings.clear();
            error = "vision encoding failed";
            return false;
        }
        tokens = n_tokens;
        dimension = n_embd;
        return true;
    } catch (const std::exception & e) {
        embeddings.clear();
        error = e.what();
        return false;
    }
}

#endif // SMOLVLA_BUILD_VISION

#if defined(SMOLVLA_BUILD_POLICY)

// policy
#include <algorithm>

struct smolvla_policy::impl {
    // Destroy the policy graphs before freeing their model-owned weights.
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<smolvla_vision> vision;
    std::unique_ptr<smolvla_prefix> prefix;
    std::unique_ptr<smolvla_expert> expert;
    std::unique_ptr<smolvla_tokenizer, decltype(&smolvla_tokenizer_free)> tokenizer{nullptr,smolvla_tokenizer_free};
};
smolvla_policy::smolvla_policy() : state(new impl) {}
smolvla_policy::~smolvla_policy() = default;
std::unique_ptr<smolvla_policy> smolvla_policy::open(const char * model, const char * mmproj, int threads, std::string & error) {
    error.clear();
    if (!model || !*model || !mmproj || !*mmproj || threads <= 0) { error="invalid SmolVLA configuration"; return nullptr; }
    auto p=std::unique_ptr<smolvla_policy>(new smolvla_policy);
    p->state->tokenizer.reset(smolvla_tokenizer_open(model));
    if (!p->state->tokenizer) { error="cannot load SmolVLA tokenizer"; return nullptr; }
    p->state->vision=smolvla_vision::open(mmproj,threads,error); if (!p->state->vision) return nullptr;
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_extra_bufts = false;
    p->state->model.reset(llama_model_load_from_file(model, params));
    if (!p->state->model) { error="cannot load SmolVLA model"; return nullptr; }
    p->state->prefix=smolvla_prefix::open(p->state->model.get(),threads,error); if (!p->state->prefix) return nullptr;
    p->state->expert=smolvla_expert::open(p->state->model.get(),threads,error); if (!p->state->expert) return nullptr;
    return p;
}
bool smolvla_policy::infer(const std::vector<smolvla_rgb_image> & images, const std::string & instruction,
                           const std::vector<float> & normalized_state, const std::vector<float> & noise,
                           std::vector<float> & actions, std::string & error) {
    actions.clear(); error.clear();
    if (images.size()!=3 || normalized_state.size()!=6 || noise.size()!=50*32) { error="expected three images, six state values and 50x32 noise"; return false; }
    int32_t ids[48]; uint8_t valid[48];
    if (smolvla_tokenizer_encode(state->tokenizer.get(),instruction.data(),instruction.size(),ids,valid)) { error="tokenization failed"; return false; }
    std::vector<std::vector<float>> visual;
    for (const auto & image:images) {
        std::vector<float> embeds; int tokens=0,dim=0;
        if (!state->vision->encode(image.data,image.width,image.height,embeds,tokens,dim,error)) return false;
        if (tokens!=64 || dim!=960) { error="unexpected vision output shape"; return false; }
        visual.push_back(std::move(embeds));
    }
    std::vector<float> padded(32,0.0f), state_embed;
    std::copy(normalized_state.begin(),normalized_state.end(),padded.begin());
    if (!state->expert->project_state(padded.data(),state_embed,error)) return false;
    std::vector<float> prefix; std::vector<uint8_t> padding,attention; std::vector<int32_t> positions;
    if (!state->prefix->assemble(visual,{ids,ids+48},{valid,valid+48},state_embed,prefix,padding,attention,positions,error)) return false;
    std::vector<std::vector<float>> keys,values;
    if (!state->prefix->forward(prefix,attention,positions,keys,values,error)) return false;
    std::vector<float> padded_actions;
    if (!state->expert->sample(noise,keys,values,padding,padded_actions,error)) return false;
    actions.resize(50*6);
    for (size_t i=0;i<50;++i) std::copy_n(padded_actions.data()+i*32,6,actions.data()+i*6);
    return true;
}

#endif // SMOLVLA_BUILD_POLICY
