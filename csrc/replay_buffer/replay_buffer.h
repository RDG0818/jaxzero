// csrc/replay_buffer/replay_buffer.h
//
// C++ prioritized experience replay buffer.
//
// Design highlights:
//  - Lock-free ring-buffer pointer (std::atomic<int64_t> write_ptr_).
//  - Lock-free sum tree for O(log N) priority sampling/updates (see sum_tree.h).
//  - Main data storage in regular malloc (78 MB, CPU-only random access).
//  - Output gather buffers (sample / reanalysis results) in CUDA pinned memory
//    when available so jax.device_put() skips the pageable→pinned copy.
//  - Two independent output buffer sets to avoid aliasing between the
//    LearnerActor (sample) and ReanalyzeActor (sample_for_reanalysis).
//  - Output buffers are lazily sized on first call and cached thereafter.
#pragma once
#include "sum_tree.h"
#include "pinned_alloc.h"
#include <atomic>
#include <cstdint>

// ---------------------------------------------------------------------------
// Configuration — mirrors ReplayBuffer.__init__ parameters.
// ---------------------------------------------------------------------------
struct ReplayBufferConfig {
    int   capacity           = 100000;
    int   obs_size           = 18;
    int   action_space_size  = 5;
    int   num_agents         = 3;
    int   unroll_steps       = 5;
    float alpha              = 0.6f;
    float beta_start         = 0.4f;
    int   beta_frames        = 100000;
};

// ---------------------------------------------------------------------------
// A set of pre-allocated (optionally pinned) output arrays for one batch type.
// All six ReplayItem fields + importance weights + indices live here.
// ---------------------------------------------------------------------------
struct PinnedBatch {
    float*   observations   = nullptr; // (B, num_agents, obs_size)
    int32_t* actions        = nullptr; // (B, unroll_steps, num_agents)
    float*   policy_targets = nullptr; // (B, unroll_steps+1, num_agents, action_space_size)
    float*   value_targets  = nullptr; // (B, unroll_steps+1, num_agents)
    float*   reward_targets = nullptr; // (B, unroll_steps, num_agents)
    int32_t* agent_orders   = nullptr; // (B, num_agents)
    float*   weights        = nullptr; // (B,)  IS weights
    int64_t* indices        = nullptr; // (B,)  buffer indices
    int      batch_size     = 0;
    bool     is_pinned      = false;
};


class ReplayBuffer {
public:
    explicit ReplayBuffer(const ReplayBufferConfig& cfg);
    ~ReplayBuffer();

    // Non-copyable; moveable only internally.
    ReplayBuffer(const ReplayBuffer&)            = delete;
    ReplayBuffer& operator=(const ReplayBuffer&) = delete;

    // ------------------------------------------------------------------
    // Core API  (mirrors Python ReplayBuffer)
    // ------------------------------------------------------------------

    /// Copy one ReplayItem into the ring buffer at the next slot.
    /// priority <= 0 → use max existing priority (or 1.0 if buffer empty).
    void add(const float*   observation,    // (num_agents, obs_size)
             const int32_t* actions,        // (unroll_steps, num_agents)
             const float*   policy_target,  // (unroll_steps+1, num_agents, action_space_size)
             const float*   value_target,   // (unroll_steps+1, num_agents)
             const float*   reward_target,  // (unroll_steps, num_agents)
             const int32_t* agent_order,    // (num_agents,)
             float          priority);

    /// Stratified PER sample.  Gathers `batch_size` items into pinned output
    /// buffers.  Returns false (and leaves output undefined) when empty.
    bool sample(int batch_size);

    /// Uniform sample without replacement (for ReanalyzeActor).
    /// Only fills observations and agent_orders in the output buffers.
    /// Returns false when empty.
    bool sample_for_reanalysis(int batch_size);

    /// Batch priority update after TD-error computation.
    void update_priorities(const int64_t* indices,
                           const float*   priorities, int n);

    /// In-place update of policy_targets[:,0] and value_targets[:,0].
    void update_targets(const int64_t* indices,
                        const float*   policy_targets,  // (n, num_agents, action_space_size)
                        const float*   root_values,     // (n,)
                        int n);

    // ------------------------------------------------------------------
    // Accessors used by pybind11 bindings
    // ------------------------------------------------------------------
    const PinnedBatch&        get_sample_out()     const { return sample_out_; }
    const PinnedBatch&        get_reanalysis_out() const { return reanalysis_out_; }
    const ReplayBufferConfig& config()             const { return cfg_; }

    int   size()         const;
    int   capacity()     const { return cfg_.capacity; }
    float current_beta() const;

    struct Stats {
        int   size, capacity;
        float fill_pct;
        float priority_min, priority_max, priority_mean, priority_std;
        float beta;
    };
    Stats get_stats() const;

private:
    ReplayBufferConfig cfg_;

    // Main storage (regular malloc, struct-of-arrays layout).
    float*   s_observations_   = nullptr; // [capacity, num_agents, obs_size]
    int32_t* s_actions_        = nullptr; // [capacity, unroll_steps, num_agents]
    float*   s_policy_targets_ = nullptr; // [capacity, unroll_steps+1, num_agents, A]
    float*   s_value_targets_  = nullptr; // [capacity, unroll_steps+1, num_agents]
    float*   s_reward_targets_ = nullptr; // [capacity, unroll_steps, num_agents]
    int32_t* s_agent_orders_   = nullptr; // [capacity, num_agents]

    // Priority tracking array for get_stats() (not used for sampling).
    float*   s_priorities_     = nullptr; // [capacity]

    SumTree              sum_tree_;
    PinnedBatch          sample_out_;      // output of sample()
    PinnedBatch          reanalysis_out_;  // output of sample_for_reanalysis()

    std::atomic<int64_t> write_ptr_{0};   // monotonic; slot = write_ptr_ % capacity
    std::atomic<int64_t> size_{0};        // saturates at capacity
    std::atomic<int64_t> frame_count_{0}; // for beta annealing
    uint64_t             rng_state_;

    // Precomputed element counts per item (for indexing and memcpy sizes).
    int obs_n_; // num_agents * obs_size
    int act_n_; // unroll_steps * num_agents
    int pol_n_; // (unroll_steps+1) * num_agents * action_space_size
    int val_n_; // (unroll_steps+1) * num_agents
    int rew_n_; // unroll_steps * num_agents
    int ord_n_; // num_agents

    void alloc_storage();
    void free_storage() noexcept;

    void ensure_sample_out(int batch_size);
    void ensure_reanalysis_out(int batch_size);
    void alloc_pinned(PinnedBatch& pb, int batch_size);
    void free_pinned(PinnedBatch& pb) noexcept;

    /// Gather all six fields for `n` items at `indices` into `pb`.
    void gather(PinnedBatch& pb, const int64_t* indices, int n) const noexcept;
    /// Gather only observations and agent_orders (for reanalysis).
    void gather_reanalysis(PinnedBatch& pb, const int64_t* indices, int n) const noexcept;

    float xorshift_uniform() noexcept;
    uint64_t xorshift_uint64() noexcept;
};
