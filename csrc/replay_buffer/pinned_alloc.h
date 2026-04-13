// csrc/replay_buffer/pinned_alloc.h
//
// Thin helpers for allocating page-locked (pinned) host memory when CUDA is
// available, falling back to regular calloc otherwise.  Pinned memory enables
// the CUDA DMA engine to transfer directly from host to device without an
// intermediate pageable copy — important for the sample() output buffers that
// are passed to jax.device_put() every training step.
#pragma once
#include <cstdlib>
#include <cstring>

#ifdef HAS_CUDA
#  include <cuda_runtime.h>
#endif

/// Allocate `bytes` of zeroed host memory.  Uses cudaMallocHost if CUDA is
/// available and the allocation succeeds; falls back to calloc.
/// Returns {ptr, was_pinned}.
inline std::pair<void*, bool> pinned_malloc(std::size_t bytes) {
#ifdef HAS_CUDA
    {
        void* ptr = nullptr;
        if (cudaMallocHost(&ptr, bytes) == cudaSuccess) {
            std::memset(ptr, 0, bytes);
            return {ptr, true};
        }
    }
    // cudaMallocHost failed (e.g. out of pinned budget) — fall through.
#endif
    return {std::calloc(1, bytes), false};
}

/// Free memory previously allocated by pinned_malloc.
inline void pinned_free(void* ptr, bool was_pinned) noexcept {
    if (!ptr) return;
#ifdef HAS_CUDA
    if (was_pinned) {
        cudaFreeHost(ptr);
        return;
    }
#endif
    std::free(ptr);
}
