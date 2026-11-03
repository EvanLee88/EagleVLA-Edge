#if defined(SMOLVLA_BUILD_LLAMA)
#include "models.h"

#include "nlohmann/json.hpp"

void llama_model_smolvla::load_arch_hparams(llama_model_loader & ml) {
    llama_model_llama::load_arch_hparams(ml);

    const int64_t schema_key = gguf_find_key(ml.metadata, "smolvla.tensor_schema_version");
    if (schema_key >= 0 && (gguf_get_kv_type(ml.metadata, schema_key) != GGUF_TYPE_UINT32 ||
                            gguf_get_val_u32(ml.metadata, schema_key) != 1)) {
        throw std::runtime_error("unsupported SmolVLA tensor schema");
    }
    const int64_t config_key = gguf_find_key(ml.metadata, "smolvla.policy_config");
    if (config_key < 0 || gguf_get_kv_type(ml.metadata, config_key) != GGUF_TYPE_STRING) {
        throw std::runtime_error("missing SmolVLA policy configuration");
    }
    const auto config = nlohmann::json::parse(gguf_get_val_str(ml.metadata, config_key));
    const auto & inputs = config.at("input_features");
    const auto & outputs = config.at("output_features");
    size_t image_count = 0;
    for (const auto & item : inputs.items()) {
        const auto & feature = item.value();
        if (feature.at("type") == "VISUAL") {
            ++image_count;
            if (feature.at("shape") != nlohmann::json::array({3, 256, 256})) {
                throw std::runtime_error("unsupported SmolVLA image shape");
            }
        }
    }
    if (image_count != 3 || inputs.at("observation.state").at("type") != "STATE" ||
        inputs.at("observation.state").at("shape") != nlohmann::json::array({6}) ||
        outputs.at("action").at("type") != "ACTION" ||
        outputs.at("action").at("shape") != nlohmann::json::array({6}) ||
        config.at("type") != "smolvla" || config.at("tokenizer_max_length") != 48 ||
        config.at("resize_imgs_with_padding") != nlohmann::json::array({512, 512}) ||
        config.at("empty_cameras") != 0 || config.at("adapt_to_pi_aloha") != false ||
        config.at("use_delta_joint_actions_aloha") != false || config.at("use_cache") != true ||
        config.at("add_image_special_tokens") != false || config.at("pad_language_to") != "max_length" ||
        config.at("prefix_length") != 0 ||
        config.at("chunk_size").get<int>() != 50 || config.at("max_action_dim").get<int>() != 32 ||
        config.at("max_state_dim").get<int>() != 32 || config.at("num_vlm_layers").get<int>() != 16 ||
        config.at("num_expert_layers").get<int>() != 0 || config.at("expert_width_multiplier").get<double>() != 0.75 ||
        config.at("num_steps").get<int>() != 10 || config.at("self_attn_every_n_layers").get<int>() != 2 ||
        config.at("attention_mode").get<std::string>() != "cross_attn" ||
        config.at("min_period").get<double>() != 0.004 || config.at("max_period").get<double>() != 4.0 ||
        hparams.n_layer() != 16 || hparams.n_embd != 960 || hparams.n_head() != 15 ||
        hparams.n_head_kv() != 5 || hparams.n_ff() != 2560 ||
        hparams.f_norm_rms_eps != 1e-5f || hparams.rope_freq_base_train != 10000.0f) {
        throw std::runtime_error("unsupported SmolVLA model configuration");
    }
}

// This first stage loads the policy weights alongside the Llama-compatible
// prefix. The existing CPU policy still owns its own execution path.
void llama_model_smolvla::load_arch_tensors(llama_model_loader & ml) {
    llama_model_llama::load_arch_tensors(ml);

    auto add = [this](const LLM_TN_IMPL & name, std::initializer_list<int64_t> shape, int flags = 0) {
        if (auto * tensor = create_tensor(name, shape, flags)) {
            policy_tensors.push_back(tensor);
        }
    };

    constexpr int64_t action = 32, hidden = 720, prefix = 960, ff = 2048, kv = 320;
    add(tn(LLM_TENSOR_SMOLVLA_ACTION_IN, "weight"), {action, hidden});
    add(tn(LLM_TENSOR_SMOLVLA_ACTION_IN, "bias"), {hidden});
    add(tn(LLM_TENSOR_SMOLVLA_ACTION_OUT, "weight"), {hidden, action});
    add(tn(LLM_TENSOR_SMOLVLA_ACTION_OUT, "bias"), {action});
    add(tn(LLM_TENSOR_SMOLVLA_TIME_IN, "weight"), {2 * hidden, hidden});
    add(tn(LLM_TENSOR_SMOLVLA_TIME_IN, "bias"), {hidden});
    add(tn(LLM_TENSOR_SMOLVLA_TIME_OUT, "weight"), {hidden, hidden});
    add(tn(LLM_TENSOR_SMOLVLA_TIME_OUT, "bias"), {hidden});
    add(tn(LLM_TENSOR_SMOLVLA_STATE, "weight"), {action, prefix});
    add(tn(LLM_TENSOR_SMOLVLA_STATE, "bias"), {prefix});
    add(tn(LLM_TENSOR_SMOLVLA_EXPERT_NORM, "weight"), {hidden});
    add(tn(LLM_TENSOR_SMOLVLA_VLM_HEAD, "weight"), {prefix, vocab.n_tokens()}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < 16; ++i) {
        const int64_t kv_input = i % 2 ? kv : hidden;
        add(tn(LLM_TENSOR_SMOLVLA_EXPERT_ATTN_NORM, "weight", i), {hidden});
        add(tn(LLM_TENSOR_SMOLVLA_EXPERT_FFN_NORM, "weight", i), {hidden});
        add(tn(LLM_TENSOR_SMOLVLA_EXPERT_Q, "weight", i), {hidden, prefix});
        add(tn(LLM_TENSOR_SMOLVLA_EXPERT_K, "weight", i), {kv_input, kv});
        add(tn(LLM_TENSOR_SMOLVLA_EXPERT_V, "weight", i), {kv_input, kv});
        add(tn(LLM_TENSOR_SMOLVLA_EXPERT_O, "weight", i), {prefix, hidden});
        add(tn(LLM_TENSOR_SMOLVLA_EXPERT_GATE, "weight", i), {hidden, ff});
        add(tn(LLM_TENSOR_SMOLVLA_EXPERT_UP, "weight", i), {hidden, ff});
        add(tn(LLM_TENSOR_SMOLVLA_EXPERT_DOWN, "weight", i), {ff, hidden});
    }
}

#endif // SMOLVLA_BUILD_LLAMA

#if defined(SMOLVLA_BUILD_CPU)
#include "jetson_pi_smolvla.h"
#include "smolvla-model.h"

// SmolVLA CPU graphs use a dedicated ggml backend scheduler. The policy may
// borrow weights from a llama model; standalone reference tools load their own.

// action input
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <map>
#include <tuple>

namespace {
constexpr int action_dim = 32, hidden = 720, steps = 50;

// Model-owned tensors are borrowed, never freed or modified here.
ggml_tensor * borrowed_tensor(const llama_model * model, const std::string & name, int64_t a, int64_t b, std::string & error) {
    const auto * t = smolvla_model_tensor(model, name.c_str());
    if (!t || t->ne[0] != a || t->ne[1] != b || t->ne[2] != 1 || t->ne[3] != 1 ||
        (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
        !t->buffer || !ggml_backend_buffer_is_host(t->buffer) || !t->data) {
        error = "invalid or non-CPU model tensor: " + name;
        return nullptr;
    }
    return const_cast<ggml_tensor *>(t);
}
struct cpu_graph_scheduler {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend{ggml_backend_cpu_init(), ggml_backend_free};
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> sched{nullptr, ggml_backend_sched_free};
    explicit cpu_graph_scheduler(int threads) {
        if (!backend) return;
        ggml_backend_cpu_set_n_threads(backend.get(), threads);
        ggml_backend_t backends[] = {backend.get()};
        sched.reset(ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false));
    }
};
// The scheduler must outlive its allocated tensors, and be destroyed before
// their metadata context and the borrowed or owned model weights.
struct reusable_graph {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init({1024 * 1024, nullptr, true}), ggml_free};
    cpu_graph_scheduler runtime;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * inputs[3] = {};
    ggml_tensor * outputs[3] = {};
    size_t shape = 0;
    explicit reusable_graph(int threads) : runtime(threads) {}
};
struct tensor_spec { const char * name; int64_t a, b; };
constexpr tensor_spec specs[] = {
    {"smolvla.action_in_proj.weight", action_dim, hidden},
    {"smolvla.action_in_proj.bias", hidden, 1},
    {"smolvla.action_time_mlp_in.weight", hidden * 2, hidden},
    {"smolvla.action_time_mlp_in.bias", hidden, 1},
    {"smolvla.action_time_mlp_out.weight", hidden, hidden},
    {"smolvla.action_time_mlp_out.bias", hidden, 1},
    {"smolvla.action_out_proj.weight", hidden, action_dim},
    {"smolvla.action_out_proj.bias", action_dim, 1},
    {"smolvla.state_proj.weight", 32, 960},
    {"smolvla.state_proj.bias", 960, 1},
    {"smolvla.vlm_with_expert.lm_expert.norm.weight", hidden, 1},
};
}

struct smolvla_action_input::impl {
    ggml_context * weights = nullptr;
    ggml_tensor * tensors[11] = {};
    int threads = 1;
    std::unique_ptr<reusable_graph> graphs[4];
    ~impl() { for (auto & graph : graphs) graph.reset(); if (weights) ggml_free(weights); }
};
smolvla_action_input::smolvla_action_input() : state(new impl) {}
smolvla_action_input::~smolvla_action_input() = default;

std::unique_ptr<smolvla_action_input> smolvla_action_input::open(const char * path, int threads, std::string & error) {
    error.clear();
    if (!path || !*path || threads <= 0) { error = "invalid action input configuration"; return nullptr; }
    ggml_context * header_ctx = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file(path, {true, &header_ctx}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> header(header_ctx, ggml_free);
    if (!file || !header) { error = "cannot read action GGUF"; return nullptr; }
    const auto get_string = [&](const char * key) -> const char * {
        const auto id = gguf_find_key(file.get(), key);
        return id >= 0 && gguf_get_kv_type(file.get(), id) == GGUF_TYPE_STRING ? gguf_get_val_str(file.get(), id) : nullptr;
    };
    const char * architecture = get_string("general.architecture");
    const char * config_text = get_string("smolvla.policy_config");
    if (!architecture || std::strcmp(architecture, "smolvla") != 0 || !config_text) {
        error = "missing SmolVLA architecture or policy configuration"; return nullptr;
    }
    const auto schema = gguf_find_key(file.get(), "smolvla.tensor_schema_version");
    if (schema >= 0 && (gguf_get_kv_type(file.get(), schema) != GGUF_TYPE_UINT32 ||
                        gguf_get_val_u32(file.get(), schema) != 1)) {
        error = "unsupported SmolVLA tensor schema"; return nullptr;
    }
    try {
        const auto config = nlohmann::json::parse(config_text);
        if (config.at("chunk_size").get<int>() != steps || config.at("max_action_dim").get<int>() != action_dim ||
            config.at("max_state_dim").get<int>() != 32 || config.at("num_vlm_layers").get<int>() != 16 ||
            config.at("num_expert_layers").get<int>() != 0 || config.at("expert_width_multiplier").get<double>() != 0.75 ||
            config.at("num_steps").get<int>() != 10 || config.at("self_attn_every_n_layers").get<int>() != 2 ||
            config.at("attention_mode").get<std::string>() != "cross_attn" ||
            config.at("min_period").get<double>() != 0.004 || config.at("max_period").get<double>() != 4.0) {
            error = "unsupported SmolVLA model configuration"; return nullptr;
        }
    } catch (const std::exception & e) { error = std::string("invalid SmolVLA policy configuration: ") + e.what(); return nullptr; }
    int64_t ids[11]; size_t bytes = 1024 * 1024;
    for (size_t i = 0; i < 11; ++i) {
        ids[i] = gguf_find_tensor(file.get(), specs[i].name);
        auto * t = ggml_get_tensor(header.get(), specs[i].name);
        if (ids[i] < 0 || !t || t->ne[0] != specs[i].a || t->ne[1] != specs[i].b || t->ne[2] != 1 || t->ne[3] != 1 ||
            (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)) {
            error = std::string("invalid action tensor: ") + specs[i].name;
            return nullptr;
        }
        bytes += ggml_nbytes(t) + ggml_tensor_overhead() + 64;
    }
    auto result = std::unique_ptr<smolvla_action_input>(new smolvla_action_input);
    result->state->threads = threads;
    result->state->weights = ggml_init({bytes, nullptr, false});
    if (!result->state->weights) { error = "cannot allocate action weights"; return nullptr; }
    std::ifstream source(path, std::ios::binary);
    if (!source) { error = "cannot open action GGUF"; return nullptr; }
    for (size_t i = 0; i < 11; ++i) {
        auto * meta = ggml_get_tensor(header.get(), specs[i].name);
        auto * dst = ggml_new_tensor(result->state->weights, meta->type, 2, meta->ne);
        result->state->tensors[i] = dst;
        const size_t offset = gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), ids[i]);
        if (offset > (size_t) std::numeric_limits<std::streamoff>::max()) { error = "invalid tensor offset"; return nullptr; }
        source.seekg((std::streamoff) offset);
        source.read(static_cast<char *>(dst->data), ggml_nbytes(dst));
        if (!source) { error = std::string("cannot read action tensor: ") + specs[i].name; return nullptr; }
    }
    return result;
}

std::unique_ptr<smolvla_action_input> smolvla_action_input::open(const llama_model * model, int threads, std::string & error) {
    error.clear();
    if (!model || threads <= 0) { error = "invalid action input configuration"; return nullptr; }
    auto result = std::unique_ptr<smolvla_action_input>(new smolvla_action_input);
    result->state->threads = threads;
    for (size_t i = 0; i < 11; ++i) {
        result->state->tensors[i] = borrowed_tensor(model, specs[i].name, specs[i].a, specs[i].b, error);
        if (!result->state->tensors[i]) return nullptr;
    }
    return result;
}

bool smolvla_action_input::encode(const float * noisy_actions, float time, std::vector<float> & output, std::string & error) {
    output.clear(); error.clear();
    if (!noisy_actions || !std::isfinite(time) || time < 0 || time > 1) { error = "invalid action input or time"; return false; }
    for (int i = 0; i < steps * action_dim; ++i) if (!std::isfinite(noisy_actions[i])) { error = "non-finite action input"; return false; }
    if (!state->graphs[0]) {
        auto candidate = std::unique_ptr<reusable_graph>(new reusable_graph(state->threads));
        auto & ctx = candidate->ctx;
        auto & runtime = candidate->runtime;
        if (!ctx) { error = "cannot allocate action graph"; return false; }
        if (!runtime.sched) { error = "cannot initialize action scheduler"; return false; }
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, action_dim, steps);
        ggml_set_input(x);
        auto linear = [&](ggml_tensor * input, int index) {
            return ggml_add(ctx.get(), ggml_mul_mat(ctx.get(), state->tensors[index], input), state->tensors[index + 1]);
        };
        auto * projected = linear(x, 0);
        auto * time_embedding = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden, steps);
        ggml_set_input(time_embedding);
        auto * combined = ggml_concat(ctx.get(), projected, time_embedding, 0);
        auto * result = linear(ggml_silu(ctx.get(), linear(combined, 2)), 4);
        ggml_set_output(result);
        auto * graph = ggml_new_graph(ctx.get());
        candidate->graph = graph;
        ggml_build_forward_expand(graph, result);
        if (!ggml_backend_sched_alloc_graph(runtime.sched.get(), graph)) { error = "cannot allocate action scheduler graph"; return false; }
        candidate->inputs[0] = x;
        candidate->inputs[1] = time_embedding;
        candidate->outputs[0] = result;
        state->graphs[0] = std::move(candidate);
    }
    auto & cached = *state->graphs[0];
    auto & runtime = cached.runtime;
    auto * graph = cached.graph;
    auto * x = cached.inputs[0];
    auto * time_embedding = cached.inputs[1];
    auto * result = cached.outputs[0];
    std::vector<float> time_values(hidden * steps);
    float * values = time_values.data();
    for (int d = 0; d < hidden / 2; ++d) {
        const double fraction = (double) d / (hidden / 2 - 1);
        const double period = 0.004 * std::pow(4.0 / 0.004, fraction);
        const double phase = (2.0 * 3.14159265358979323846 / period) * (double) time;
        for (int j = 0; j < steps; ++j) {
            values[j * hidden + d] = (float) std::sin(phase);
            values[j * hidden + d + hidden / 2] = (float) std::cos(phase);
        }
    }

    ggml_backend_tensor_set(x, noisy_actions, 0, steps * action_dim * sizeof(float));
    ggml_backend_tensor_set(time_embedding, time_values.data(), 0, time_values.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(runtime.sched.get(), graph) != GGML_STATUS_SUCCESS) {
        state->graphs[0].reset(); error = "action graph computation failed"; return false;
    }
    output.resize(steps * hidden);
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    return true;
}


bool smolvla_action_input::project_velocity(const float * expert_output, std::vector<float> & velocity, std::string & error) {
    velocity.clear(); error.clear();
    if (!expert_output) { error = "invalid expert output"; return false; }
    for (int i = 0; i < steps * hidden; ++i) {
        if (!std::isfinite(expert_output[i])) { error = "non-finite expert output"; return false; }
    }
    if (!state->graphs[1]) {
        auto candidate = std::unique_ptr<reusable_graph>(new reusable_graph(state->threads));
        auto & ctx = candidate->ctx;
        auto & runtime = candidate->runtime;
        if (!ctx) { error = "cannot allocate velocity graph"; return false; }
        if (!runtime.sched) { error = "cannot initialize velocity scheduler"; return false; }
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden, steps);
        ggml_set_input(x);
        auto * result = ggml_add(ctx.get(), ggml_mul_mat(ctx.get(), state->tensors[6], x), state->tensors[7]);
        ggml_set_output(result);
        auto * graph = ggml_new_graph(ctx.get());
        candidate->graph = graph;
        ggml_build_forward_expand(graph, result);
        if (!ggml_backend_sched_alloc_graph(runtime.sched.get(), graph)) { error = "cannot allocate velocity scheduler graph"; return false; }
        candidate->inputs[0] = x;
        candidate->outputs[0] = result;
        state->graphs[1] = std::move(candidate);
    }
    auto & cached = *state->graphs[1];
    auto & runtime = cached.runtime;
    auto * graph = cached.graph;
    auto * x = cached.inputs[0];
    auto * result = cached.outputs[0];

    ggml_backend_tensor_set(x, expert_output, 0, steps * hidden * sizeof(float));
    if (ggml_backend_sched_graph_compute(runtime.sched.get(), graph) != GGML_STATUS_SUCCESS) {
        state->graphs[1].reset(); error = "velocity graph computation failed"; return false;
    }
    velocity.resize(steps * action_dim);
    ggml_backend_tensor_get(result, velocity.data(), 0, velocity.size() * sizeof(float));
    return true;
}


bool smolvla_action_input::project_state(const float * state_input, std::vector<float> & embedding, std::string & error) {
    embedding.clear(); error.clear();
    if (!state_input) { error = "invalid state input"; return false; }
    for (int i = 0; i < 32; ++i) {
        if (!std::isfinite(state_input[i])) { error = "non-finite state input"; return false; }
    }
    if (!state->graphs[2]) {
        auto candidate = std::unique_ptr<reusable_graph>(new reusable_graph(state->threads));
        auto & ctx = candidate->ctx;
        auto & runtime = candidate->runtime;
        if (!ctx) { error = "cannot allocate state graph"; return false; }
        if (!runtime.sched) { error = "cannot initialize state scheduler"; return false; }
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 1);
        ggml_set_input(x);
        auto * result = ggml_add(ctx.get(), ggml_mul_mat(ctx.get(), state->tensors[8], x), state->tensors[9]);
        ggml_set_output(result);
        auto * graph = ggml_new_graph(ctx.get());
        candidate->graph = graph;
        ggml_build_forward_expand(graph, result);
        if (!ggml_backend_sched_alloc_graph(runtime.sched.get(), graph)) { error = "cannot allocate state scheduler graph"; return false; }
        candidate->inputs[0] = x;
        candidate->outputs[0] = result;
        state->graphs[2] = std::move(candidate);
    }
    auto & cached = *state->graphs[2];
    auto & runtime = cached.runtime;
    auto * graph = cached.graph;
    auto * x = cached.inputs[0];
    auto * result = cached.outputs[0];

    ggml_backend_tensor_set(x, state_input, 0, 32 * sizeof(float));
    if (ggml_backend_sched_graph_compute(runtime.sched.get(), graph) != GGML_STATUS_SUCCESS) {
        state->graphs[2].reset(); error = "state graph computation failed"; return false;
    }
    embedding.resize(960);
    ggml_backend_tensor_get(result, embedding.data(), 0, embedding.size() * sizeof(float));
    return true;
}


bool smolvla_action_input::normalize_expert(const float * expert_hidden, std::vector<float> & output, std::string & error) {
    output.clear(); error.clear();
    if (!expert_hidden) { error = "invalid expert hidden state"; return false; }
    for (int i = 0; i < steps * hidden; ++i) {
        if (!std::isfinite(expert_hidden[i])) { error = "non-finite expert hidden state"; return false; }
    }
    if (!state->graphs[3]) {
        auto candidate = std::unique_ptr<reusable_graph>(new reusable_graph(state->threads));
        auto & ctx = candidate->ctx;
        auto & runtime = candidate->runtime;
        if (!ctx) { error = "cannot allocate expert norm graph"; return false; }
        if (!runtime.sched) { error = "cannot initialize expert norm scheduler"; return false; }
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden, steps);
        ggml_set_input(x);
        auto * norm = ggml_rms_norm(ctx.get(), x, 1e-5f);
        // LlamaRMSNorm casts the normalized activation back to the input dtype
        // before multiplying by its BF16 weight. Both operations round to BF16.
        norm = ggml_cast(ctx.get(), ggml_cast(ctx.get(), norm, GGML_TYPE_BF16), GGML_TYPE_F32);
        auto * result = ggml_mul(ctx.get(), norm, state->tensors[10]);
        result = ggml_cast(ctx.get(), ggml_cast(ctx.get(), result, GGML_TYPE_BF16), GGML_TYPE_F32);
        ggml_set_output(result);
        auto * graph = ggml_new_graph(ctx.get());
        candidate->graph = graph;
        ggml_build_forward_expand(graph, result);
        if (!ggml_backend_sched_alloc_graph(runtime.sched.get(), graph)) { error = "cannot allocate expert norm scheduler graph"; return false; }
        candidate->inputs[0] = x;
        candidate->outputs[0] = result;
        state->graphs[3] = std::move(candidate);
    }
    auto & cached = *state->graphs[3];
    auto & runtime = cached.runtime;
    auto * graph = cached.graph;
    auto * x = cached.inputs[0];
    auto * result = cached.outputs[0];

    ggml_backend_tensor_set(x, expert_hidden, 0, steps * hidden * sizeof(float));
    if (ggml_backend_sched_graph_compute(runtime.sched.get(), graph) != GGML_STATUS_SUCCESS) {
        state->graphs[3].reset(); error = "expert norm graph computation failed"; return false;
    }
    output.resize(steps * hidden);
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    return true;
}


// rope
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <memory>

namespace {
// A bounded per-thread cache keeps independent scheduler state for callers on
// different threads. Shapes and thread count determine the complete topology.
struct rope_graph {
    size_t tokens, heads;
    int threads;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{nullptr, ggml_free};
    cpu_graph_scheduler runtime;
    ggml_tensor * x = nullptr, * positions = nullptr, * result = nullptr;
    ggml_cgraph * graph = nullptr;

    rope_graph(size_t n, size_t h, int n_threads) : tokens(n), heads(h), threads(n_threads), runtime(n_threads) {
        ctx.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!ctx || !runtime.sched) return;
        x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 64, heads, tokens);
        positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
        ggml_set_input(x); ggml_set_input(positions);
        result = ggml_rope_ext(ctx.get(), x, positions, nullptr, 64, GGML_ROPE_TYPE_NEOX,
                               0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_set_output(result);
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, result);
        if (!ggml_backend_sched_alloc_graph(runtime.sched.get(), graph)) graph = nullptr;
    }
};
}

namespace smolvla {
bool expert_rope(const float * input, size_t tokens, size_t heads, const int32_t * positions,
                 int threads, std::vector<float> & output, std::string & error) {
    output.clear(); error.clear();
    if (!input || !positions || !tokens || tokens > 8192 || !heads || heads > 64 || threads <= 0) {
        error = "invalid RoPE input"; return false;
    }
    const size_t count = tokens * heads * 64;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(input[i])) { error = "non-finite RoPE input"; return false; }
    }
    using key_type = std::tuple<size_t, size_t, int>;
    static thread_local std::map<key_type, std::unique_ptr<rope_graph>> cache;
    const key_type key{tokens, heads, threads};
    auto it = cache.find(key);
    if (it == cache.end()) {
        if (cache.size() >= 8) cache.clear();
        auto entry = std::make_unique<rope_graph>(tokens, heads, threads);
        if (!entry->graph) { error = "cannot allocate RoPE scheduler graph"; return false; }
        it = cache.emplace(key, std::move(entry)).first;
    }
    auto & entry = *it->second;
    ggml_backend_tensor_set(entry.x, input, 0, count * sizeof(float));
    ggml_backend_tensor_set(entry.positions, positions, 0, tokens * sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(entry.runtime.sched.get(), entry.graph) != GGML_STATUS_SUCCESS) {
        cache.erase(it); error = "RoPE graph failed"; return false;
    }
    output.resize(count);
    ggml_backend_tensor_get(entry.result, output.data(), 0, count * sizeof(float));
    return true;
}
}


// attention
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace {
struct attention_graph {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{nullptr, ggml_free};
    cpu_graph_scheduler runtime;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * mask = nullptr, * result = nullptr;
    ggml_cgraph * graph = nullptr;

    attention_graph(size_t nq, size_t nk, int threads, bool round_probs) : runtime(threads) {
        ctx.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!ctx || !runtime.sched) return;
        q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 64, 15, nq);
        k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 64, 15, nk);
        v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 64, 15, nk);
        mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, nk, nq);
        ggml_set_input(q); ggml_set_input(k); ggml_set_input(v); ggml_set_input(mask);
        auto * qp = ggml_permute(ctx.get(), q, 0, 2, 1, 3);
        auto * kp = ggml_permute(ctx.get(), k, 0, 2, 1, 3);
        auto * vp = ggml_cont(ctx.get(), ggml_permute(ctx.get(), v, 1, 2, 0, 3));
        auto * scores = ggml_mul_mat(ctx.get(), kp, qp);
        auto * probs = ggml_soft_max_ext(ctx.get(), scores, mask, 0.125f, 0.0f);
        if (round_probs) probs = ggml_cast(ctx.get(), ggml_cast(ctx.get(), probs, GGML_TYPE_BF16), GGML_TYPE_F32);
        result = ggml_mul_mat(ctx.get(), vp, probs);
        result = ggml_cont_2d(ctx.get(), ggml_permute(ctx.get(), result, 0, 2, 1, 3), 15 * 64, nq);
        ggml_set_output(result);
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, result);
        if (!ggml_backend_sched_alloc_graph(runtime.sched.get(), graph)) graph = nullptr;
    }
};
}

namespace smolvla {
bool attention(const float * query, const float * key, const float * value, size_t nq, size_t nk,
               const uint8_t * mask, int threads, std::vector<float> & output, std::string & error, bool bf16_probabilities) {
    output.clear(); error.clear();
    if (!query || !key || !value || !mask || !nq || !nk || nq > 1024 || nk > 1024 || threads <= 0) {
        error = "invalid attention input"; return false;
    }
    for (size_t i = 0; i < nq * 15 * 64; ++i) if (!std::isfinite(query[i])) { error = "invalid query"; return false; }
    for (size_t i = 0; i < nk * 5 * 64; ++i) if (!std::isfinite(key[i]) || !std::isfinite(value[i])) { error = "invalid key/value"; return false; }
    for (size_t i = 0; i < nq * nk; ++i) if (mask[i] > 1) { error = "invalid attention mask"; return false; }
    using key_type = std::tuple<size_t, size_t, int, bool>;
    static thread_local std::map<key_type, std::unique_ptr<attention_graph>> cache;
    const key_type cache_key{nq, nk, threads, bf16_probabilities};
    auto it = cache.find(cache_key);
    if (it == cache.end()) {
        if (cache.size() >= 4) cache.clear();
        auto entry = std::make_unique<attention_graph>(nq, nk, threads, bf16_probabilities);
        if (!entry->graph) { error = "cannot allocate attention scheduler graph"; return false; }
        it = cache.emplace(cache_key, std::move(entry)).first;
    }
    auto & entry = *it->second;
    std::vector<float> expanded_k(nk * 15 * 64), expanded_v(nk * 15 * 64), attention_mask(nq * nk);
    float * kd = expanded_k.data(), * vd = expanded_v.data();
    for (size_t t = 0; t < nk; ++t) for (size_t h = 0; h < 15; ++h) {
        std::memcpy(kd + (t * 15 + h) * 64, key + (t * 5 + h / 3) * 64, 64 * sizeof(float));
        std::memcpy(vd + (t * 15 + h) * 64, value + (t * 5 + h / 3) * 64, 64 * sizeof(float));
    }
    float * md = attention_mask.data();
    // LeRobot uses float32 minimum (not -infinity) for all masked positions.
    // When a query is entirely masked its softmax is therefore uniform.
    for (size_t i = 0; i < nq * nk; ++i) md[i] = mask[i] ? 0.0f : -std::numeric_limits<float>::max();
    ggml_backend_tensor_set(entry.q, query, 0, nq * 15 * 64 * sizeof(float));
    ggml_backend_tensor_set(entry.k, expanded_k.data(), 0, expanded_k.size() * sizeof(float));
    ggml_backend_tensor_set(entry.v, expanded_v.data(), 0, expanded_v.size() * sizeof(float));
    ggml_backend_tensor_set(entry.mask, attention_mask.data(), 0, attention_mask.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(entry.runtime.sched.get(), entry.graph) != GGML_STATUS_SUCCESS) {
        cache.erase(it); error = "attention graph failed"; return false;
    }
    output.resize(nq * 15 * 64);
    ggml_backend_tensor_get(entry.result, output.data(), 0, output.size() * sizeof(float));
    return true;
}
}


// expert mlp
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>

struct smolvla_expert_mlp::impl {
    ggml_context * weights = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr, * attn_out = nullptr, * post_norm = nullptr;
    ggml_tensor * input_norm = nullptr, * q = nullptr, * k = nullptr, * v = nullptr;
    bool cross = false;
    int layer = 0;
    int threads = 1;
    std::unique_ptr<reusable_graph> graphs[3];
    ~impl() { for (auto & graph : graphs) graph.reset(); if (weights) ggml_free(weights); }
};
smolvla_expert_mlp::smolvla_expert_mlp() : state(new impl) {}
smolvla_expert_mlp::~smolvla_expert_mlp() = default;

std::unique_ptr<smolvla_expert_mlp> smolvla_expert_mlp::open(const char * path, int layer, int threads, std::string & error) {
    error.clear();
    if (!path || !*path || layer < 0 || layer >= 16 || threads <= 0) { error = "invalid expert MLP configuration"; return nullptr; }
    ggml_context * header_ptr = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file(path, {true, &header_ptr}), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> header(header_ptr, ggml_free);
    if (!file || !header) { error = "cannot read GGUF"; return nullptr; }
    const auto arch = gguf_find_key(file.get(), "general.architecture");
    if (arch < 0 || gguf_get_kv_type(file.get(), arch) != GGUF_TYPE_STRING ||
        std::strcmp(gguf_get_val_str(file.get(), arch), "smolvla") != 0) { error = "not a SmolVLA GGUF"; return nullptr; }
    const std::string base = "smolvla.vlm_with_expert.lm_expert.layers." + std::to_string(layer) + ".mlp.";
    const std::string layer_base = "smolvla.vlm_with_expert.lm_expert.layers." + std::to_string(layer) + ".";
    const std::string names[] = {base + "gate_proj.weight", base + "up_proj.weight", base + "down_proj.weight", layer_base + "self_attn.o_proj.weight", layer_base + "post_attention_layernorm.weight", layer_base + "input_layernorm.weight", layer_base + "self_attn.q_proj.weight", layer_base + "self_attn.k_proj.weight", layer_base + "self_attn.v_proj.weight"};
    const int64_t expected_a[] = {720, 720, 2048, 960, 720, 720, 720, layer % 2 ? 320 : 720, layer % 2 ? 320 : 720};
    const int64_t expected_b[] = {2048, 2048, 720, 720, 1, 1, 960, 320, 320};
    int64_t ids[9]; size_t bytes = 1024 * 1024;
    for (int i = 0; i < 9; ++i) {
        ids[i] = gguf_find_tensor(file.get(), names[i].c_str());
        auto * t = ggml_get_tensor(header.get(), names[i].c_str());
        if (ids[i] < 0 || !t || t->ne[0] != expected_a[i] || t->ne[1] != expected_b[i] ||
            t->ne[2] != 1 || t->ne[3] != 1 || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)) {
            error = "invalid expert MLP tensor: " + names[i]; return nullptr;
        }
        bytes += ggml_nbytes(t) + ggml_tensor_overhead() + 64;
    }
    auto result = std::unique_ptr<smolvla_expert_mlp>(new smolvla_expert_mlp);
    result->state->threads = threads;
    result->state->cross = layer % 2 != 0;
    result->state->layer = layer;
    result->state->weights = ggml_init({bytes, nullptr, false});
    if (!result->state->weights) { error = "cannot allocate expert MLP weights"; return nullptr; }
    std::ifstream source(path, std::ios::binary);
    if (!source) { error = "cannot open GGUF"; return nullptr; }
    ggml_tensor ** destinations[] = {&result->state->gate, &result->state->up, &result->state->down, &result->state->attn_out, &result->state->post_norm, &result->state->input_norm, &result->state->q, &result->state->k, &result->state->v};
    for (int i = 0; i < 9; ++i) {
        auto * meta = ggml_get_tensor(header.get(), names[i].c_str());
        *destinations[i] = ggml_new_tensor(result->state->weights, meta->type, 2, meta->ne);
        const size_t offset = gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), ids[i]);
        if (offset > (size_t) std::numeric_limits<std::streamoff>::max()) { error = "invalid GGUF tensor offset"; return nullptr; }
        source.seekg((std::streamoff) offset);
        source.read(static_cast<char *>((*destinations[i])->data), ggml_nbytes(*destinations[i]));
        if (!source) { error = "cannot read expert MLP tensor"; return nullptr; }
    }
    return result;
}

std::unique_ptr<smolvla_expert_mlp> smolvla_expert_mlp::open(const llama_model * model, int layer, int threads, std::string & error) {
    error.clear();
    if (!model || layer < 0 || layer >= 16 || threads <= 0) { error = "invalid expert MLP configuration"; return nullptr; }
    auto result = std::unique_ptr<smolvla_expert_mlp>(new smolvla_expert_mlp);
    result->state->threads = threads;
    result->state->cross = layer % 2 != 0;
    result->state->layer = layer;
    const std::string base = "smolvla.vlm_with_expert.lm_expert.layers." + std::to_string(layer) + ".";
    const char * names[] = {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight", "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"};
    const int64_t a[] = {720, 720, 2048, 960, 720, 720, 720, layer % 2 ? 320 : 720, layer % 2 ? 320 : 720};
    const int64_t b[] = {2048, 2048, 720, 720, 1, 1, 960, 320, 320};
    ggml_tensor ** dst[] = {&result->state->gate, &result->state->up, &result->state->down, &result->state->attn_out, &result->state->post_norm, &result->state->input_norm, &result->state->q, &result->state->k, &result->state->v};
    for (int i = 0; i < 9; ++i) {
        *dst[i] = borrowed_tensor(model, base + names[i], a[i], b[i], error);
        if (!*dst[i]) return nullptr;
    }
    return result;
}

bool smolvla_expert_mlp::run(const float * input, std::vector<float> & output, std::string & error) {
    output.clear(); error.clear();
    if (!input) { error = "missing expert MLP input"; return false; }
    for (int i = 0; i < 50 * 720; ++i) if (!std::isfinite(input[i])) { error = "non-finite expert MLP input"; return false; }
    if (!state->graphs[0]) {
        auto candidate = std::unique_ptr<reusable_graph>(new reusable_graph(state->threads));
        auto & ctx = candidate->ctx;
        auto & runtime = candidate->runtime;
        if (!ctx) { error = "cannot allocate expert MLP graph"; return false; }
        if (!runtime.sched) { error = "cannot initialize expert MLP scheduler"; return false; }
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 720, 50);
        ggml_set_input(x);
        auto round_bf16 = [&](ggml_tensor * t) { return ggml_cast(ctx.get(), ggml_cast(ctx.get(), t, GGML_TYPE_BF16), GGML_TYPE_F32); };
        auto * gate = round_bf16(ggml_mul_mat(ctx.get(), state->gate, x));
        auto * up = round_bf16(ggml_mul_mat(ctx.get(), state->up, x));
        gate = round_bf16(ggml_silu(ctx.get(), gate));
        auto * intermediate = round_bf16(ggml_mul(ctx.get(), gate, up));
        auto * result = round_bf16(ggml_mul_mat(ctx.get(), state->down, intermediate));
        ggml_set_output(result);
        auto * graph = ggml_new_graph(ctx.get());
        candidate->graph = graph;
        ggml_build_forward_expand(graph, result);
        if (!ggml_backend_sched_alloc_graph(runtime.sched.get(), graph)) { error = "cannot allocate expert MLP scheduler graph"; return false; }
        candidate->inputs[0] = x;
        candidate->outputs[0] = result;
        state->graphs[0] = std::move(candidate);
    }
    auto & cached = *state->graphs[0];
    auto & runtime = cached.runtime;
    auto * graph = cached.graph;
    auto * x = cached.inputs[0];
    auto * result = cached.outputs[0];

    ggml_backend_tensor_set(x, input, 0, 50 * 720 * sizeof(float));
    if (ggml_backend_sched_graph_compute(runtime.sched.get(), graph) != GGML_STATUS_SUCCESS) { state->graphs[0].reset(); error = "expert MLP graph failed"; return false; }
    output.resize(50 * 720);
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    return true;
}


bool smolvla_expert_mlp::finish_layer(const float * hidden, const float * attention, std::vector<float> & output, std::string & error) {
    output.clear(); error.clear();
    if (!hidden || !attention) { error = "missing expert layer input"; return false; }
    for (int i = 0; i < 50 * 720; ++i) if (!std::isfinite(hidden[i])) { error = "non-finite expert hidden state"; return false; }
    for (int i = 0; i < 50 * 960; ++i) if (!std::isfinite(attention[i])) { error = "non-finite attention output"; return false; }
    if (!state->graphs[1]) {
        auto candidate = std::unique_ptr<reusable_graph>(new reusable_graph(state->threads));
        auto & ctx = candidate->ctx;
        auto & runtime = candidate->runtime;
        if (!ctx) { error = "cannot allocate expert layer graph"; return false; }
        if (!runtime.sched) { error = "cannot initialize expert layer scheduler"; return false; }
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 720, 50);
        auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 960, 50);
        ggml_set_input(x); ggml_set_input(a);
        auto * attention_input = a;
        auto round_bf16 = [&](ggml_tensor * t) { return ggml_cast(ctx.get(), ggml_cast(ctx.get(), t, GGML_TYPE_BF16), GGML_TYPE_F32); };
        // Reference casts attention to the projection's BF16 type, then performs
        // in-place BF16 residual addition.
        a = round_bf16(a);
        auto * residual = round_bf16(ggml_add(ctx.get(), round_bf16(ggml_mul_mat(ctx.get(), state->attn_out, a)), x));
        auto * norm = round_bf16(ggml_rms_norm(ctx.get(), residual, 1e-5f));
        norm = round_bf16(ggml_mul(ctx.get(), norm, state->post_norm));
        auto * gate = round_bf16(ggml_mul_mat(ctx.get(), state->gate, norm));
        auto * up = round_bf16(ggml_mul_mat(ctx.get(), state->up, norm));
        auto * intermediate = round_bf16(ggml_mul(ctx.get(), round_bf16(ggml_silu(ctx.get(), gate)), up));
        auto * mlp = round_bf16(ggml_mul_mat(ctx.get(), state->down, intermediate));
        auto * result = round_bf16(ggml_add(ctx.get(), mlp, residual));
        ggml_set_output(result);
        auto * graph = ggml_new_graph(ctx.get());
        candidate->graph = graph;
        ggml_build_forward_expand(graph, result);
        if (!ggml_backend_sched_alloc_graph(runtime.sched.get(), graph)) { error = "cannot allocate expert layer scheduler graph"; return false; }
        candidate->inputs[0] = x;
        candidate->inputs[1] = attention_input;
        candidate->outputs[0] = result;
        state->graphs[1] = std::move(candidate);
    }
    auto & cached = *state->graphs[1];
    auto & runtime = cached.runtime;
    auto * graph = cached.graph;
    auto * x = cached.inputs[0];
    auto * attention_input = cached.inputs[1];
    auto * result = cached.outputs[0];

    ggml_backend_tensor_set(x, hidden, 0, 50 * 720 * sizeof(float));
    ggml_backend_tensor_set(attention_input, attention, 0, 50 * 960 * sizeof(float));
    if (ggml_backend_sched_graph_compute(runtime.sched.get(), graph) != GGML_STATUS_SUCCESS) { state->graphs[1].reset(); error = "expert layer graph failed"; return false; }
    output.resize(50 * 720);
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * sizeof(float));
    return true;
}


bool smolvla_expert_mlp::project_qkv(const float * hidden, const float * cache_k, const float * cache_v,
                                     size_t cache_tokens, std::vector<float> & q, std::vector<float> & k,
                                     std::vector<float> & v, std::string & error) {
    q.clear(); k.clear(); v.clear(); error.clear();
    if (!hidden || (state->cross && (!cache_k || !cache_v || !cache_tokens || cache_tokens > 1024))) {
        error = "invalid expert QKV input"; return false;
    }
    for (int i = 0; i < 50 * 720; ++i) if (!std::isfinite(hidden[i])) { error = "non-finite expert hidden state"; return false; }
    const size_t kv_tokens = state->cross ? cache_tokens : 50;
    if (state->cross) for (size_t i = 0; i < kv_tokens * 320; ++i) {
        if (!std::isfinite(cache_k[i]) || !std::isfinite(cache_v[i])) { error = "non-finite prefix cache"; return false; }
    }
    if (!state->graphs[2] || state->graphs[2]->shape != kv_tokens) {
        auto candidate = std::unique_ptr<reusable_graph>(new reusable_graph(state->threads));
        auto & ctx = candidate->ctx;
        auto & runtime = candidate->runtime;
        if (!ctx) { error = "cannot allocate expert QKV graph"; return false; }
        if (!runtime.sched) { error = "cannot initialize expert QKV scheduler"; return false; }
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 720, 50);
        ggml_set_input(x);
        auto round_bf16 = [&](ggml_tensor * t) { return ggml_cast(ctx.get(), ggml_cast(ctx.get(), t, GGML_TYPE_BF16), GGML_TYPE_F32); };
        // Input layernorm follows the incoming dtype: first expert layer receives
        // F32 suffix embeddings; later layers receive BF16 hidden states.
        auto * norm = ggml_rms_norm(ctx.get(), x, 1e-5f);
        if (state->layer != 0) norm = round_bf16(norm);
        auto * n = round_bf16(ggml_mul(ctx.get(), norm, state->input_norm));
        auto * query = round_bf16(ggml_mul_mat(ctx.get(), state->q, n));
        ggml_tensor * ki = n, * vi = n;
        if (state->cross) {
            ki = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 320, kv_tokens);
            vi = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 320, kv_tokens);
            ggml_set_input(ki); ggml_set_input(vi);
        }
        auto * keys = ggml_mul_mat(ctx.get(), state->k, ki);
        auto * values = ggml_mul_mat(ctx.get(), state->v, vi);
        if (!state->cross) { keys = round_bf16(keys); values = round_bf16(values); }
        ggml_set_output(query); ggml_set_output(keys); ggml_set_output(values);
        auto * graph = ggml_new_graph(ctx.get());
        candidate->graph = graph;
        ggml_build_forward_expand(graph, query); ggml_build_forward_expand(graph, keys); ggml_build_forward_expand(graph, values);
        if (!ggml_backend_sched_alloc_graph(runtime.sched.get(), graph)) { error = "cannot allocate expert QKV scheduler graph"; return false; }
        candidate->inputs[0] = x;
        candidate->inputs[1] = ki;
        candidate->inputs[2] = vi;
        candidate->outputs[0] = query;
        candidate->outputs[1] = keys;
        candidate->outputs[2] = values;
        candidate->shape = kv_tokens;
        state->graphs[2] = std::move(candidate);
    }
    auto & cached = *state->graphs[2];
    auto & runtime = cached.runtime;
    auto * graph = cached.graph;
    auto * x = cached.inputs[0];
    auto * ki = cached.inputs[1];
    auto * vi = cached.inputs[2];
    auto * query = cached.outputs[0];
    auto * keys = cached.outputs[1];
    auto * values = cached.outputs[2];

    ggml_backend_tensor_set(x, hidden, 0, 50 * 720 * sizeof(float));
    if (state->cross) {
        ggml_backend_tensor_set(ki, cache_k, 0, kv_tokens * 320 * sizeof(float));
        ggml_backend_tensor_set(vi, cache_v, 0, kv_tokens * 320 * sizeof(float));
    }
    if (ggml_backend_sched_graph_compute(runtime.sched.get(), graph) != GGML_STATUS_SUCCESS) { state->graphs[2].reset(); error = "expert QKV graph failed"; return false; }
    q.resize(50 * 960); k.resize(kv_tokens * 320); v.resize(kv_tokens * 320);
    ggml_backend_tensor_get(query, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_get(keys, k.data(), 0, k.size() * sizeof(float));
    ggml_backend_tensor_get(values, v.data(), 0, v.size() * sizeof(float));
    return true;
}


bool smolvla_expert_mlp::forward(const float * hidden, const float * prefix_k, const float * prefix_v,
                                  const std::vector<uint8_t> & prefix_valid, std::vector<float> & output,
                                  std::string & error) {
    output.clear();
    if (!hidden || !prefix_k || !prefix_v || prefix_valid.empty()) { error = "invalid expert layer input"; return false; }
    std::vector<uint8_t> self_mask, cross_mask;
    std::vector<int32_t> positions;
    if (!smolvla_input::build_suffix_attention(prefix_valid, 50, self_mask, cross_mask, positions, error)) return false;
    std::vector<float> q, k, v;
    if (!project_qkv(hidden, prefix_k, prefix_v, prefix_valid.size(), q, k, v, error)) return false;
    if (state->cross) {
        const int32_t first = positions[0];
        for (int32_t & position : positions) position -= first;
    }
    std::vector<float> rotated_q;
    if (!smolvla::expert_rope(q.data(), 50, 15, positions.data(), state->threads, rotated_q, error)) return false;
    // RoPE returns the original query dtype (BF16).
    for (float & x : rotated_q) x = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
    if (!state->cross) {
        std::vector<float> rotated_k;
        if (!smolvla::expert_rope(k.data(), 50, 5, positions.data(), state->threads, rotated_k, error)) return false;
        for (float & x : rotated_k) x = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
        std::vector<float> full_k(prefix_k, prefix_k + prefix_valid.size() * 320);
        std::vector<float> full_v(prefix_v, prefix_v + prefix_valid.size() * 320);
        full_k.insert(full_k.end(), rotated_k.begin(), rotated_k.end());
        full_v.insert(full_v.end(), v.begin(), v.end());
        k.swap(full_k); v.swap(full_v);
    }
    std::vector<float> attn;
    const auto & mask = state->cross ? cross_mask : self_mask;
    if (!smolvla::attention(rotated_q.data(), k.data(), v.data(), 50, k.size() / 320, mask.data(),
                            state->threads, attn, error, !state->cross)) return false;
    if (!state->cross) for (float & x : attn) x = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
    return finish_layer(hidden, attn.data(), output, error);
}


// expert

struct smolvla_expert::impl {
    std::vector<std::unique_ptr<smolvla_expert_mlp>> layers;
    std::unique_ptr<smolvla_action_input> projections;
};
smolvla_expert::smolvla_expert() : state(new impl) {}
smolvla_expert::~smolvla_expert() = default;
std::unique_ptr<smolvla_expert> smolvla_expert::open(const char * path, int threads, std::string & error) {
    auto model = std::unique_ptr<smolvla_expert>(new smolvla_expert);
    model->state->projections = smolvla_action_input::open(path, threads, error);
    if (!model->state->projections) return nullptr;
    for (int i = 0; i < 16; ++i) {
        auto layer = smolvla_expert_mlp::open(path, i, threads, error);
        if (!layer) return nullptr;
        model->state->layers.push_back(std::move(layer));
    }
    return model;
}
std::unique_ptr<smolvla_expert> smolvla_expert::open(const llama_model * source, int threads, std::string & error) {
    auto model = std::unique_ptr<smolvla_expert>(new smolvla_expert);
    model->state->projections = smolvla_action_input::open(source, threads, error);
    if (!model->state->projections) return nullptr;
    for (int i = 0; i < 16; ++i) {
        auto layer = smolvla_expert_mlp::open(source, i, threads, error);
        if (!layer) return nullptr;
        model->state->layers.push_back(std::move(layer));
    }
    return model;
}

bool smolvla_expert::project_state(const float * input, std::vector<float> & embedding, std::string & error) {
    return state->projections->project_state(input, embedding, error);
}

bool smolvla_expert::forward(const float * suffix, const std::vector<std::vector<float>> & keys,
                             const std::vector<std::vector<float>> & values, const std::vector<uint8_t> & mask,
                             std::vector<float> & output, std::string & error) {
    output.clear(); error.clear();
    if (!suffix || mask.empty() || keys.size() != 16 || values.size() != 16) { error = "invalid expert input"; return false; }
    for (size_t i = 0; i < 16; ++i) {
        if (keys[i].size() != mask.size() * 320 || values[i].size() != mask.size() * 320) { error = "invalid expert KV cache shape"; return false; }
    }
    std::vector<float> hidden(suffix, suffix + 50 * 720), next;
    for (size_t i = 0; i < state->layers.size(); ++i) {
        if (!state->layers[i]->forward(hidden.data(), keys[i].data(), values[i].data(), mask, next, error)) return false;
        hidden.swap(next);
    }
    return state->projections->normalize_expert(hidden.data(), output, error);
}


bool smolvla_expert::sample(const std::vector<float> & noise, const std::vector<std::vector<float>> & keys,
                            const std::vector<std::vector<float>> & values, const std::vector<uint8_t> & mask,
                            std::vector<float> & actions, std::string & error) {
    if (noise.size() != 50 * 32) { actions.clear(); error = "invalid action noise shape"; return false; }
    auto velocity = [&](const std::vector<float> & current, float time, std::vector<float> & result, std::string & err) {
        std::vector<float> suffix, hidden;
        if (!state->projections->encode(current.data(), time, suffix, err)) return false;
        if (!forward(suffix.data(), keys, values, mask, hidden, err)) return false;
        return state->projections->project_velocity(hidden.data(), result, err);
    };
    return smolvla::euler_integrate(noise, 10, velocity, actions, error);
}


// prefix
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>

namespace {
constexpr int layers = 16;
struct layer_weights { ggml_tensor * norm, * q, * k, * v, * out, * ffn_norm, * gate, * up, * down; };
void unpermute(float * data, size_t tokens, size_t heads) {
    for (size_t t=0;t<tokens;++t) for (size_t h=0;h<heads;++h) {
        float tmp[64]; float * row=data+(t*heads+h)*64;
        for (int i=0;i<32;++i) { tmp[i]=row[2*i]; tmp[i+32]=row[2*i+1]; }
        std::memcpy(row,tmp,sizeof(tmp));
    }
}
float bf16(float x) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)); }
}
struct smolvla_prefix::impl {
    ggml_context * weights = nullptr;
    layer_weights w[layers]{};
    ggml_tensor * token_embedding = nullptr;
    int threads = 1;

    // A scheduler owns its graph allocation; destroy it before the graph context
    // and destroy all cached graphs before releasing their referenced weights.
    struct cached_graph {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{nullptr, ggml_free};
        cpu_graph_scheduler runtime;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * input2 = nullptr;
        ggml_tensor * output = nullptr;
        ggml_tensor * output2 = nullptr;
        ggml_tensor * output3 = nullptr;
        size_t n = 0;
        explicit cached_graph(int threads) : runtime(threads) {}
    };
    std::unique_ptr<cached_graph> embedding;
    std::unique_ptr<cached_graph> qkv[layers];
    std::unique_ptr<cached_graph> residual[layers];
    size_t forward_n = 0;
    ~impl() {
        embedding.reset();
        for (int i = 0; i < layers; ++i) { qkv[i].reset(); residual[i].reset(); }
        if (weights) ggml_free(weights);
    }
};
smolvla_prefix::smolvla_prefix() : state(new impl) {}
smolvla_prefix::~smolvla_prefix() = default;

std::unique_ptr<smolvla_prefix> smolvla_prefix::open(const char * path, int threads, std::string & error) {
    error.clear();
    if (!path || !*path || threads <= 0) { error="invalid prefix configuration"; return nullptr; }
    ggml_context * meta_ptr=nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file(path,{true,&meta_ptr}),gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> meta(meta_ptr,ggml_free);
    if (!file || !meta) { error="cannot read prefix GGUF"; return nullptr; }
    const int64_t arch=gguf_find_key(file.get(),"general.architecture");
    if (arch<0 || gguf_get_kv_type(file.get(),arch)!=GGUF_TYPE_STRING || std::strcmp(gguf_get_val_str(file.get(),arch),"smolvla")) { error="not a SmolVLA GGUF"; return nullptr; }
    const int64_t rope=gguf_find_key(file.get(),"smolvla.rope.freq_base");
    if (rope<0 || gguf_get_kv_type(file.get(),rope)!=GGUF_TYPE_FLOAT32 || gguf_get_val_f32(file.get(),rope)!=10000.0f) { error="invalid SmolVLA RoPE metadata; reconvert model"; return nullptr; }
    const char * names[]={"attn_norm.weight","attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight","ffn_norm.weight","ffn_gate.weight","ffn_up.weight","ffn_down.weight"};
    const int64_t a[]={960,960,960,960,960,960,960,960,2560};
    const int64_t b[]={1,960,320,320,960,1,2560,2560,960};
    size_t bytes=1024*1024;
    for(int l=0;l<layers;++l) for(int j=0;j<9;++j) {
        const std::string name="blk."+std::to_string(l)+"."+names[j];
        auto * t=ggml_get_tensor(meta.get(),name.c_str());
        if(!t || t->ne[0]!=a[j] || t->ne[1]!=b[j] || t->ne[2]!=1 || t->ne[3]!=1 ||
           (t->type!=GGML_TYPE_F32 && t->type!=GGML_TYPE_F16 && t->type!=GGML_TYPE_BF16)) { error="invalid prefix tensor: "+name; return nullptr; }
        bytes+=ggml_nbytes(t)+ggml_tensor_overhead()+64;
    }
    auto * token_meta = ggml_get_tensor(meta.get(), "token_embd.weight");
    const auto token_id = gguf_find_tensor(file.get(), "token_embd.weight");
    if (!token_meta || token_id < 0 || token_meta->ne[0] != 960 || token_meta->ne[1] != 49280 ||
        token_meta->ne[2] != 1 || token_meta->ne[3] != 1 ||
        (token_meta->type != GGML_TYPE_F32 && token_meta->type != GGML_TYPE_F16 && token_meta->type != GGML_TYPE_BF16)) {
        error = "invalid token embedding tensor"; return nullptr;
    }
    bytes += ggml_nbytes(token_meta) + ggml_tensor_overhead() + 64;
    auto result=std::unique_ptr<smolvla_prefix>(new smolvla_prefix);
    result->state->threads=threads; result->state->weights=ggml_init({bytes,nullptr,false});
    if(!result->state->weights) { error="cannot allocate prefix weights"; return nullptr; }
    std::ifstream in(path,std::ios::binary); if(!in) { error="cannot open prefix GGUF"; return nullptr; }
    result->state->token_embedding = ggml_new_tensor(result->state->weights, token_meta->type, 2, token_meta->ne);
    const size_t token_offset = gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), token_id);
    if (token_offset > (size_t)std::numeric_limits<std::streamoff>::max()) { error="invalid token embedding offset"; return nullptr; }
    in.seekg((std::streamoff)token_offset);
    in.read(static_cast<char *>(result->state->token_embedding->data), ggml_nbytes(result->state->token_embedding));
    if (!in) { error="cannot read token embedding"; return nullptr; }
    for(int l=0;l<layers;++l) {
        auto & w=result->state->w[l]; ggml_tensor ** dst[]={&w.norm,&w.q,&w.k,&w.v,&w.out,&w.ffn_norm,&w.gate,&w.up,&w.down};
        for(int j=0;j<9;++j) {
            const std::string name="blk."+std::to_string(l)+"."+names[j];
            auto * t=ggml_get_tensor(meta.get(),name.c_str()); const auto id=gguf_find_tensor(file.get(),name.c_str());
            *dst[j]=ggml_new_tensor(result->state->weights,t->type,2,t->ne);
            size_t offset=gguf_get_data_offset(file.get())+gguf_get_tensor_offset(file.get(),id);
            if(offset>(size_t)std::numeric_limits<std::streamoff>::max()) { error="invalid prefix tensor offset"; return nullptr; }
            in.seekg((std::streamoff)offset); in.read(static_cast<char *>((*dst[j])->data),ggml_nbytes(*dst[j]));
            if(!in) { error="cannot read prefix tensor: "+name; return nullptr; }
        }
    }
    return result;
}

std::unique_ptr<smolvla_prefix> smolvla_prefix::open(const llama_model * model, int threads, std::string & error) {
    error.clear();
    if (!model || threads <= 0) { error = "invalid prefix configuration"; return nullptr; }
    auto result = std::unique_ptr<smolvla_prefix>(new smolvla_prefix);
    result->state->threads = threads;
    result->state->token_embedding = borrowed_tensor(model, "token_embd.weight", 960, 49280, error);
    if (!result->state->token_embedding) return nullptr;
    const char * names[] = {"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
    const int64_t a[] = {960,960,960,960,960,960,960,960,2560};
    const int64_t b[] = {1,960,320,320,960,1,2560,2560,960};
    for (int l = 0; l < layers; ++l) {
        auto & w = result->state->w[l];
        ggml_tensor ** dst[] = {&w.norm,&w.q,&w.k,&w.v,&w.out,&w.ffn_norm,&w.gate,&w.up,&w.down};
        for (int j = 0; j < 9; ++j) {
            *dst[j] = borrowed_tensor(model, "blk." + std::to_string(l) + "." + names[j], a[j], b[j], error);
            if (!*dst[j]) return nullptr;
        }
    }
    return result;
}

bool smolvla_prefix::embed_tokens(const std::vector<int32_t> & tokens, std::vector<float> & output, std::string & error) {
    output.clear(); error.clear();
    if (tokens.empty() || tokens.size() > 1024) { error="invalid token count"; return false; }
    for (int32_t token : tokens) if (token < 0 || token >= 49280) { error="invalid token ID"; return false; }
    if (!state->embedding || state->embedding->n != tokens.size()) {
        // Discard the old allocation first to bound peak backend memory.
        state->embedding.reset();
        auto cache = std::unique_ptr<impl::cached_graph>(new impl::cached_graph(state->threads));
        cache->ctx.reset(ggml_init({1024*1024,nullptr,true}));
        if (!cache->ctx) { error="cannot allocate token embedding graph"; return false; }
        if (!cache->runtime.sched) { error="cannot initialize token embedding scheduler"; return false; }
        auto * ctx=cache->ctx.get();
        cache->input=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,tokens.size());
        ggml_set_input(cache->input);
        cache->output=ggml_get_rows(ctx,state->token_embedding,cache->input);
        ggml_set_output(cache->output);
        cache->graph=ggml_new_graph(ctx); ggml_build_forward_expand(cache->graph,cache->output);
        if (!ggml_backend_sched_alloc_graph(cache->runtime.sched.get(),cache->graph)) { error="cannot allocate token embedding scheduler graph"; return false; }
        cache->n=tokens.size();
        state->embedding=std::move(cache);
    }
    auto & cache=*state->embedding;
    ggml_backend_tensor_set(cache.input,tokens.data(),0,tokens.size()*sizeof(int32_t));
    if (ggml_backend_sched_graph_compute(cache.runtime.sched.get(),cache.graph)!=GGML_STATUS_SUCCESS) {
        state->embedding.reset(); error="token embedding graph failed"; return false;
    }
    output.resize(tokens.size()*960);
    ggml_backend_tensor_get(cache.output,output.data(),0,output.size()*sizeof(float));
    const float * src=output.data();
    // LeRobot multiplies BF16 token embeddings by sqrt(hidden_size), producing BF16.
    const float scale=std::sqrt(960.0f);
    for(size_t i=0;i<output.size();++i) output[i]=bf16(src[i]*scale);
    return true;
}

bool smolvla_prefix::assemble(const std::vector<std::vector<float>> & images, const std::vector<int32_t> & tokens,
                              const std::vector<uint8_t> & language_valid, const std::vector<float> & state_embedding,
                              std::vector<float> & prefix, std::vector<uint8_t> & padding,
                              std::vector<uint8_t> & attention, std::vector<int32_t> & positions, std::string & error) {
    prefix.clear(); padding.clear(); attention.clear(); positions.clear(); error.clear();
    if (images.size()!=3 || tokens.size()!=48 || language_valid.size()!=48 || state_embedding.size()!=960) {
        error="invalid prefix assembly dimensions"; return false;
    }
    for (const auto & image:images) {
        if (image.size()!=64*960) { error="invalid image embedding dimensions"; return false; }
        for (float x:image) if (!std::isfinite(x)) { error="non-finite image embedding"; return false; }
    }
    for (float x:state_embedding) if (!std::isfinite(x)) { error="non-finite state embedding"; return false; }
    std::vector<float> language;
    if (!embed_tokens(tokens,language,error)) return false;
    if (!smolvla_input::build_prefix_attention({1,1,1},64,language_valid,padding,attention,positions,error)) return false;
    prefix.reserve((3*64+48+1)*960);
    const float scale=std::sqrt(960.0f);
    for (const auto & image:images) for (float x:image) prefix.push_back(x*scale);
    prefix.insert(prefix.end(),language.begin(),language.end());
    prefix.insert(prefix.end(),state_embedding.begin(),state_embedding.end());
    return true;
}

bool smolvla_prefix::forward(const std::vector<float> & prefix, const std::vector<uint8_t> & mask,
                             const std::vector<int32_t> & positions, std::vector<std::vector<float>> & keys,
                             std::vector<std::vector<float>> & values, std::string & error) {
    keys.clear(); values.clear(); error.clear();
    const size_t n=positions.size();
    if(!n || n>1024 || prefix.size()!=n*960 || mask.size()!=n*n) { error="invalid prefix input shape"; return false; }
    for(float x:prefix) if(!std::isfinite(x)) { error="non-finite prefix"; return false; }
    if (state->forward_n != n) {
        for (int i=0; i<layers; ++i) { state->qkv[i].reset(); state->residual[i].reset(); }
        state->forward_n=n;
    }
    std::vector<float> hidden=prefix;
    std::vector<std::vector<float>> next_keys, next_values;
    for(int l=0;l<layers;++l) {
        auto & w=state->w[l];
        if (!state->qkv[l] || state->qkv[l]->n != n) {
            state->qkv[l].reset();
            auto cache=std::unique_ptr<impl::cached_graph>(new impl::cached_graph(state->threads));
            cache->ctx.reset(ggml_init({1024*1024,nullptr,true}));
            if(!cache->ctx) { error="cannot allocate prefix graph"; return false; }
            if(!cache->runtime.sched) { error="cannot initialize prefix scheduler"; return false; }
            auto * ctx=cache->ctx.get();
            cache->input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,960,n); ggml_set_input(cache->input);
            auto round=[&](ggml_tensor * t) { return ggml_cast(ctx,ggml_cast(ctx,t,GGML_TYPE_BF16),GGML_TYPE_F32); };
            auto * norm=ggml_rms_norm(ctx,cache->input,1e-5f);
            if(l>0) norm=round(norm);
            norm=round(ggml_mul(ctx,norm,w.norm));
            cache->output=round(ggml_mul_mat(ctx,w.q,norm));
            cache->output2=round(ggml_mul_mat(ctx,w.k,norm));
            cache->output3=round(ggml_mul_mat(ctx,w.v,norm));
            ggml_set_output(cache->output); ggml_set_output(cache->output2); ggml_set_output(cache->output3);
            cache->graph=ggml_new_graph(ctx);
            ggml_build_forward_expand(cache->graph,cache->output); ggml_build_forward_expand(cache->graph,cache->output2); ggml_build_forward_expand(cache->graph,cache->output3);
            if(!ggml_backend_sched_alloc_graph(cache->runtime.sched.get(),cache->graph)) { error="cannot allocate prefix QKV scheduler graph"; return false; }
            cache->n=n; state->qkv[l]=std::move(cache);
        }
        auto & qkv=*state->qkv[l];
        ggml_backend_tensor_set(qkv.input,hidden.data(),0,hidden.size()*sizeof(float));
        if(ggml_backend_sched_graph_compute(qkv.runtime.sched.get(),qkv.graph)!=GGML_STATUS_SUCCESS) {
            state->qkv[l].reset(); error="prefix QKV failed"; return false;
        }
        std::vector<float> qv(n*960),kv(n*320),vv(n*320),qr,kr;
        ggml_backend_tensor_get(qkv.output,qv.data(),0,qv.size()*sizeof(float)); ggml_backend_tensor_get(qkv.output2,kv.data(),0,kv.size()*sizeof(float)); ggml_backend_tensor_get(qkv.output3,vv.data(),0,vv.size()*sizeof(float));
        unpermute(qv.data(),n,15); unpermute(kv.data(),n,5);
        if(!smolvla::expert_rope(qv.data(),n,15,positions.data(),state->threads,qr,error) ||
           !smolvla::expert_rope(kv.data(),n,5,positions.data(),state->threads,kr,error)) return false;
        for(float & z:qr) z=bf16(z);
        for(float & z:kr) z=bf16(z);
        next_keys.push_back(kr); next_values.push_back(vv);
        std::vector<float> attn;
        if(!smolvla::attention(qr.data(),kr.data(),vv.data(),n,n,mask.data(),state->threads,attn,error,true)) return false;
        for(float & z:attn) z=bf16(z);
        if (!state->residual[l] || state->residual[l]->n != n) {
            state->residual[l].reset();
            auto cache=std::unique_ptr<impl::cached_graph>(new impl::cached_graph(state->threads));
            cache->ctx.reset(ggml_init({1024*1024,nullptr,true}));
            if(!cache->ctx) { error="cannot allocate prefix residual graph"; return false; }
            if(!cache->runtime.sched) { error="cannot initialize prefix residual scheduler"; return false; }
            auto * ctx=cache->ctx.get();
            cache->input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,960,n);
            cache->input2=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,960,n);
            ggml_set_input(cache->input); ggml_set_input(cache->input2);
            auto rnd=[&](ggml_tensor * t) { return ggml_cast(ctx,ggml_cast(ctx,t,GGML_TYPE_BF16),GGML_TYPE_F32); };
            auto * residual=rnd(ggml_add(ctx,rnd(ggml_mul_mat(ctx,w.out,cache->input2)),cache->input));
            auto * z=rnd(ggml_rms_norm(ctx,residual,1e-5f)); z=rnd(ggml_mul(ctx,z,w.ffn_norm));
            auto * gate=rnd(ggml_mul_mat(ctx,w.gate,z)); auto * up=rnd(ggml_mul_mat(ctx,w.up,z));
            auto * inner=rnd(ggml_mul(ctx,rnd(ggml_silu(ctx,gate)),up));
            cache->output=rnd(ggml_add(ctx,rnd(ggml_mul_mat(ctx,w.down,inner)),residual));
            ggml_set_output(cache->output);
            cache->graph=ggml_new_graph(ctx); ggml_build_forward_expand(cache->graph,cache->output);
            if(!ggml_backend_sched_alloc_graph(cache->runtime.sched.get(),cache->graph)) { error="cannot allocate prefix residual scheduler graph"; return false; }
            cache->n=n; state->residual[l]=std::move(cache);
        }
        auto & residual=*state->residual[l];
        ggml_backend_tensor_set(residual.input,hidden.data(),0,hidden.size()*sizeof(float));
        ggml_backend_tensor_set(residual.input2,attn.data(),0,attn.size()*sizeof(float));
        if(ggml_backend_sched_graph_compute(residual.runtime.sched.get(),residual.graph)!=GGML_STATUS_SUCCESS) {
            state->residual[l].reset(); error="prefix residual graph failed"; return false;
        }
        ggml_backend_tensor_get(residual.output,hidden.data(),0,hidden.size()*sizeof(float));
    }
    keys.swap(next_keys); values.swap(next_values);
    return true;
}


#endif // SMOLVLA_BUILD_CPU
