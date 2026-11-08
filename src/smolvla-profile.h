#pragma once

// NVTX v3 is header-only. The disabled build neither includes nor links NVTX.
#if defined(SMOLVLA_NVTX)
#include <nvtx3/nvToolsExt.h>
#include <atomic>
#include <cstdint>
#endif

namespace smolvla_profile {

class scoped_range {
public:
#if defined(SMOLVLA_NVTX)
    explicit scoped_range(const char * label) noexcept { nvtxRangePushA(label); }
    scoped_range(const char * label, std::uint64_t id) noexcept {
        nvtxEventAttributes_t event{};
        event.version = NVTX_VERSION;
        event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        event.messageType = NVTX_MESSAGE_TYPE_ASCII;
        event.message.ascii = label;
        event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        event.payload.ullValue = id;
        nvtxRangePushEx(&event);
    }
    ~scoped_range() noexcept { nvtxRangePop(); }
#else
    explicit scoped_range(const char *) noexcept {}
    scoped_range(const char *, unsigned long long) noexcept {}
    ~scoped_range() noexcept = default;
#endif
    scoped_range(const scoped_range &) = delete;
    scoped_range & operator=(const scoped_range &) = delete;
};

#if defined(SMOLVLA_NVTX)
inline std::uint64_t next_inference_id() noexcept {
    static std::atomic<std::uint64_t> id{0};
    return id.fetch_add(1, std::memory_order_relaxed) + 1;
}
#endif

} // namespace smolvla_profile

#define SMOLVLA_PROFILE_JOIN_IMPL(a, b) a##b
#define SMOLVLA_PROFILE_JOIN(a, b) SMOLVLA_PROFILE_JOIN_IMPL(a, b)
#if defined(SMOLVLA_NVTX)
#define SMOLVLA_PROFILE_SCOPE(label) \
    ::smolvla_profile::scoped_range SMOLVLA_PROFILE_JOIN(smolvla_profile_range_, __LINE__){label}
#define SMOLVLA_PROFILE_SCOPE_ID(label, id) \
    ::smolvla_profile::scoped_range SMOLVLA_PROFILE_JOIN(smolvla_profile_range_, __LINE__){label, static_cast<std::uint64_t>(id)}
#else
// Do not evaluate label/id expressions, allocate labels, or increment IDs when disabled.
#define SMOLVLA_PROFILE_SCOPE(label) ((void) 0)
#define SMOLVLA_PROFILE_SCOPE_ID(label, id) ((void) 0)
#endif
