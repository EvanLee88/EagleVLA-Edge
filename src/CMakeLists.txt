llama_add_compile_flags()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/pi0-build-config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/pi0-build-config.h
    @ONLY
)

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-kv-cache-dsv4.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            jetson_pi_smolvla.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_compile_definitions(llama PRIVATE SMOLVLA_BUILD_TOKENIZER SMOLVLA_BUILD_LLAMA)
target_include_directories(llama PRIVATE . ../vendor)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()

if (LLAMA_BUILD_COMMON AND (LLAMA_BUILD_TOOLS OR JETSON_PI_BUILD_MTMD))
    add_library(jetson_pi_smolvla_input STATIC jetson_pi_smolvla.cpp ../include/jetson_pi_smolvla.h)
    target_compile_definitions(jetson_pi_smolvla_input PRIVATE SMOLVLA_BUILD_INPUT)
    target_include_directories(jetson_pi_smolvla_input PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_link_libraries(jetson_pi_smolvla_input PUBLIC ggml)
    target_compile_features(jetson_pi_smolvla_input PUBLIC cxx_std_17)
    set_target_properties(jetson_pi_smolvla_input PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (LLAMA_BUILD_TESTS)
        add_executable(smolvla-preprocess-reference ../tools/smolvla-reference/preprocess.cpp)
        target_link_libraries(smolvla-preprocess-reference PRIVATE jetson_pi_smolvla_input)
        add_executable(smolvla-prefix-mask-reference ../tools/smolvla-reference/prefix_mask.cpp)
        target_link_libraries(smolvla-prefix-mask-reference PRIVATE jetson_pi_smolvla_input)
        add_executable(smolvla-suffix-mask-reference ../tools/smolvla-reference/suffix_mask.cpp)
        target_link_libraries(smolvla-suffix-mask-reference PRIVATE jetson_pi_smolvla_input)
        add_executable(smolvla-flow-reference ../tools/smolvla-reference/flow.cpp)
        target_link_libraries(smolvla-flow-reference PRIVATE jetson_pi_smolvla_input)
    endif()

    if (GGML_CPU AND NOT GGML_BACKEND_DL)
        add_library(jetson_pi_smolvla_action_input STATIC models/smolvla.cpp ../include/jetson_pi_smolvla.h)
        target_compile_definitions(jetson_pi_smolvla_action_input PRIVATE SMOLVLA_BUILD_CPU)
        target_include_directories(jetson_pi_smolvla_action_input PUBLIC ../include PRIVATE . ../vendor)
        target_link_libraries(jetson_pi_smolvla_action_input PUBLIC ggml ggml-cpu PRIVATE jetson_pi_smolvla_input llama)
        target_compile_features(jetson_pi_smolvla_action_input PUBLIC cxx_std_17)
        set_target_properties(jetson_pi_smolvla_action_input PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if (LLAMA_BUILD_TESTS)
            add_executable(smolvla-action-input-reference ../tools/smolvla-reference/action_input.cpp)
            target_link_libraries(smolvla-action-input-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-action-output-reference ../tools/smolvla-reference/action_output.cpp)
            target_link_libraries(smolvla-action-output-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-state-reference ../tools/smolvla-reference/state.cpp)
            target_link_libraries(smolvla-state-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-expert-norm-reference ../tools/smolvla-reference/expert_norm.cpp)
            target_link_libraries(smolvla-expert-norm-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-expert-mlp-reference ../tools/smolvla-reference/expert_mlp.cpp)
            target_link_libraries(smolvla-expert-mlp-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-expert-finish-reference ../tools/smolvla-reference/expert_finish.cpp)
            target_link_libraries(smolvla-expert-finish-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-expert-qkv-reference ../tools/smolvla-reference/expert_qkv.cpp)
            target_link_libraries(smolvla-expert-qkv-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-expert-layer-reference ../tools/smolvla-reference/expert_layer.cpp)
            target_link_libraries(smolvla-expert-layer-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-expert-reference ../tools/smolvla-reference/expert.cpp)
            target_link_libraries(smolvla-expert-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-prefix-reference ../tools/smolvla-reference/prefix.cpp)
            target_link_libraries(smolvla-prefix-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-assemble-reference ../tools/smolvla-reference/assemble_prefix.cpp)
            target_link_libraries(smolvla-assemble-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-sample-reference ../tools/smolvla-reference/sample.cpp)
            target_link_libraries(smolvla-sample-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-rope-reference ../tools/smolvla-reference/rope.cpp)
            target_link_libraries(smolvla-rope-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(smolvla-attention-reference ../tools/smolvla-reference/attention.cpp)
            target_link_libraries(smolvla-attention-reference PRIVATE jetson_pi_smolvla_action_input)
            add_executable(test-smolvla-graph-reuse ../tests/test_smolvla_graph_reuse.cpp)
            target_link_libraries(test-smolvla-graph-reuse PRIVATE jetson_pi_smolvla_action_input)
        endif()

    endif()

    add_library(jetson_pi_smolvla_vision STATIC jetson_pi_smolvla.cpp ../include/jetson_pi_smolvla.h)
    target_compile_definitions(jetson_pi_smolvla_vision PRIVATE SMOLVLA_BUILD_VISION)
    target_include_directories(jetson_pi_smolvla_vision PUBLIC ../include PRIVATE ../tools/mtmd)
    target_link_libraries(jetson_pi_smolvla_vision PUBLIC mtmd PRIVATE jetson_pi_smolvla_input)
    target_compile_features(jetson_pi_smolvla_vision PUBLIC cxx_std_17)
    set_target_properties(jetson_pi_smolvla_vision PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (GGML_CPU AND NOT GGML_BACKEND_DL)
        add_library(jetson_pi_smolvla_policy STATIC jetson_pi_smolvla.cpp ../include/jetson_pi_smolvla.h)
        target_compile_definitions(jetson_pi_smolvla_policy PRIVATE SMOLVLA_BUILD_POLICY)
        target_include_directories(jetson_pi_smolvla_policy PUBLIC ../include)
        target_link_libraries(jetson_pi_smolvla_policy PUBLIC jetson_pi_smolvla_vision jetson_pi_smolvla_action_input llama)
        target_compile_features(jetson_pi_smolvla_policy PUBLIC cxx_std_17)
        set_target_properties(jetson_pi_smolvla_policy PROPERTIES POSITION_INDEPENDENT_CODE ON)
        if (LLAMA_BUILD_TESTS)
            add_executable(smolvla-policy-reference ../tools/smolvla-reference/policy.cpp)
            target_link_libraries(smolvla-policy-reference PRIVATE jetson_pi_smolvla_policy)
        endif()
    endif()
    if (LLAMA_BUILD_TESTS)
        add_executable(smolvla-vision-reference ../tools/smolvla-reference/vision.cpp)
        target_link_libraries(smolvla-vision-reference PRIVATE jetson_pi_smolvla_vision)
    endif()

    add_library(jetson_pi_gr00t_input STATIC
        gr00t-input.h
        gr00t-input.cpp
    )
    target_include_directories(jetson_pi_gr00t_input PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(jetson_pi_gr00t_input PUBLIC cxx_std_17)
    set_target_properties(jetson_pi_gr00t_input PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(jetson_pi_pi0
        ../include/jetson_pi_pi0.h
        jetson_pi_pi0.cpp
    )
    add_library(jetson_pi_llm
        ../include/jetson_pi_llm.h
        jetson_pi_llm.cpp
    )
    add_library(jetson_pi_mllm
        ../include/jetson_pi_mllm.h
        jetson_pi_mllm.cpp
    )
    add_library(jetson_pi_gr00t
        ../include/jetson_pi_gr00t.h
        jetson_pi_gr00t.cpp
    )

    foreach(target jetson_pi_pi0 jetson_pi_llm jetson_pi_mllm jetson_pi_gr00t)
        set_target_properties(${target} PROPERTIES
            VERSION ${LLAMA_INSTALL_VERSION}
            SOVERSION 0
            INSTALL_RPATH "$ORIGIN"
        )
        target_include_directories(${target} PUBLIC ../include)
        target_compile_features(${target} PRIVATE cxx_std_17)
        if (NOT BUILD_SHARED_LIBS)
            set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
    endforeach()

    target_link_libraries(jetson_pi_pi0 PUBLIC mtmd llama ggml)
    target_link_libraries(jetson_pi_llm PUBLIC llama ggml)
    target_link_libraries(jetson_pi_mllm PUBLIC mtmd llama ggml)
    target_link_libraries(jetson_pi_gr00t PUBLIC mtmd llama ggml PRIVATE jetson_pi_gr00t_input)

    install(TARGETS jetson_pi_pi0 jetson_pi_llm jetson_pi_mllm jetson_pi_gr00t
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/jetson_pi_pi0.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/jetson_pi_llm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/jetson_pi_mllm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/jetson_pi_gr00t.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()
