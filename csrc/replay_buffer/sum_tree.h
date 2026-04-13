// csrc/replay_buffer/sum_tree.h
//
// Lock-free binary segment tree (sum tree) for prioritized experience replay.
//
// Nodes are stored as std::atomic<uint32_t> and reinterpreted as IEEE-754
// float via bit_cast.  Internal nodes hold the sum of their subtree's alpha-
// scaled priorities.  All atomic operations use memory_order_relaxed because
// the ReplayBufferActor is a single Ray actor — all calls are serialized at the
// Ray level (SPSC).  The atomics prevent UB and are ready for multi-producer
// use with the relaxed→acq_rel upgrade.
//
// Tree layout (1-indexed):
//   node[1]   = root (sum of all priorities)
//   node[2i]  = left child of node[i]
//   node[2i+1]= right child of node[i]
//   leaves    = node[N_] … node[2*N_ - 1]   (N_ = next power-of-two >= capacity)
//   slot s    → leaf node[N_ + s]
#pragma once
#include <atomic>
#include <bit>        // std::bit_cast  (C++20)
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>     // std::unique_ptr, std::make_unique
#include <cassert>

// ---------------------------------------------------------------------------
// Portable float <-> uint32_t bit reinterpretation (avoids UB from union/ptr).
// ---------------------------------------------------------------------------
inline float    bits_to_float(uint32_t bits) noexcept { return std::bit_cast<float>(bits); }
inline uint32_t float_to_bits(float    f)    noexcept { return std::bit_cast<uint32_t>(f); }

// ---------------------------------------------------------------------------
// Float-safe atomic add via CAS loop.
// In the SPSC case this never spins — the compare always succeeds on the
// first try since no other thread is modifying the node concurrently.
// ---------------------------------------------------------------------------
static void float_atomic_add(std::atomic<uint32_t>& node, float delta) noexcept {
    uint32_t old_bits = node.load(std::memory_order_relaxed);
    uint32_t new_bits;
    do {
        new_bits = float_to_bits(bits_to_float(old_bits) + delta);
    } while (!node.compare_exchange_weak(
                 old_bits, new_bits,
                 std::memory_order_relaxed,
                 std::memory_order_relaxed));
}

// ---------------------------------------------------------------------------
// xorshift64 — fast non-cryptographic PRNG, returns value in [0, 1).
// ---------------------------------------------------------------------------
inline float xorshift_uniform(uint64_t& state) noexcept {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    // Map to [0, 1) by taking the top 23 mantissa bits.
    uint32_t mantissa = static_cast<uint32_t>(state >> 41);
    uint32_t bits = (127u << 23) | mantissa;
    return bits_to_float(bits) - 1.0f;
}


class SumTree {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    SumTree() = default;

    SumTree(int capacity, float alpha)
        : capacity_(capacity), alpha_(alpha)
    {
        // Round capacity up to the next power of two so all leaves are at the
        // same depth and the parent formula (i>>1) is exact.
        N_ = 1;
        while (N_ < capacity) N_ <<= 1;

        // std::atomic is neither copyable nor moveable, so std::vector can't
        // be used (its reallocation path needs move/copy semantics).
        // Use a unique_ptr to a raw array instead.
        tree_size_ = 2 * N_;
        nodes_ = std::make_unique<std::atomic<uint32_t>[]>(tree_size_);
        for (int i = 0; i < tree_size_; ++i)
            nodes_[i].store(0u, std::memory_order_relaxed);
        max_priority_bits_.store(0u, std::memory_order_relaxed);
    }

    // -----------------------------------------------------------------------
    // Update the priority of slot `slot` to `new_priority`.
    // O(log N) — one exchange on the leaf + one CAS-loop per ancestor.
    // -----------------------------------------------------------------------
    void update(int slot, float new_priority) noexcept {
        assert(slot >= 0 && slot < capacity_);
        float alpha_p = std::pow(new_priority, alpha_);

        int leaf = N_ + slot;
        uint32_t old_bits = nodes_[leaf].exchange(
            float_to_bits(alpha_p), std::memory_order_relaxed);
        float delta = alpha_p - bits_to_float(old_bits);

        // Propagate delta up to the root.
        for (int node = leaf >> 1; node >= 1; node >>= 1)
            float_atomic_add(nodes_[node], delta);

        // Track running maximum (for default add-priority logic).
        update_max(new_priority);
    }

    // -----------------------------------------------------------------------
    // Stratified PER sampling: divides [0, total) into B equal segments and
    // draws one sample per segment.  Fills indices[B] and probs[B].
    // probs[i] = alpha-scaled priority / total (caller computes IS weights).
    // -----------------------------------------------------------------------
    void sample(int B, int cur_size, int* indices, float* probs,
                uint64_t& rng_state) const noexcept {
        float total = bits_to_float(nodes_[1].load(std::memory_order_relaxed));
        if (total <= 0.0f || cur_size == 0) {
            for (int i = 0; i < B; ++i) { indices[i] = 0; probs[i] = 1.0f; }
            return;
        }
        float segment = total / static_cast<float>(B);

        for (int i = 0; i < B; ++i) {
            float target = (static_cast<float>(i) + xorshift_uniform(rng_state))
                           * segment;

            // Prefix-sum tree walk from root to a leaf.
            int node = 1;
            while (node < N_) {
                int left = 2 * node;
                float left_val = bits_to_float(
                    nodes_[left].load(std::memory_order_relaxed));
                if (left_val >= target) {
                    node = left;
                } else {
                    target -= left_val;
                    node = left + 1;
                }
            }
            int slot = node - N_;
            // Guard: uninitialized leaves beyond cur_size have priority 0 and
            // may be reached if floating-point rounding pushes target slightly
            // past the last initialised leaf's prefix sum.
            if (slot >= cur_size) slot = cur_size - 1;
            indices[i] = slot;
            float p = bits_to_float(nodes_[node].load(std::memory_order_relaxed));
            probs[i] = (p > 0.0f) ? p / total : 1e-8f / total;
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    float total()        const noexcept {
        return bits_to_float(nodes_[1].load(std::memory_order_relaxed));
    }
    float max_priority() const noexcept {
        return bits_to_float(max_priority_bits_.load(std::memory_order_relaxed));
    }

private:
    int   capacity_  = 0;
    int   N_         = 0;   // tree leaf count (power-of-two)
    int   tree_size_ = 0;   // = 2 * N_
    float alpha_     = 0.6f;

    std::unique_ptr<std::atomic<uint32_t>[]> nodes_;
    std::atomic<uint32_t> max_priority_bits_{0};

    // CAS loop to atomically track the maximum priority ever seen.
    // In SPSC this fires at most once per unique maximum (very rarely).
    void update_max(float candidate) noexcept {
        uint32_t cand_bits = float_to_bits(candidate);
        uint32_t cur = max_priority_bits_.load(std::memory_order_relaxed);
        while (bits_to_float(cur) < candidate) {
            if (max_priority_bits_.compare_exchange_weak(
                    cur, cand_bits,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed))
                break;
            // cur was refreshed by the failed CAS; the while condition re-checks.
        }
    }
};
