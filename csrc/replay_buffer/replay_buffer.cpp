// csrc/replay_buffer/replay_buffer.cpp
#include "replay_buffer.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

ReplayBuffer::ReplayBuffer(const ReplayBufferConfig& cfg)
    : cfg_(cfg),
      sum_tree_(cfg.capacity, cfg.alpha),
      rng_state_(0xdeadbeefcafe1234ULL)
{
    if (cfg.capacity <= 0)
        throw std::invalid_argument("ReplayBuffer: capacity must be > 0");

    obs_n_ = cfg.num_agents * cfg.obs_size;
    act_n_ = cfg.unroll_steps * cfg.num_agents;
    pol_n_ = (cfg.unroll_steps + 1) * cfg.num_agents * cfg.action_space_size;
    val_n_ = (cfg.unroll_steps + 1) * cfg.num_agents;
    rew_n_ = cfg.unroll_steps * cfg.num_agents;
    ord_n_ = cfg.num_agents;

    alloc_storage();
}

ReplayBuffer::~ReplayBuffer() {
    free_storage();
    free_pinned(sample_out_);
    free_pinned(reanalysis_out_);
}

// ---------------------------------------------------------------------------
// Storage allocation / deallocation
// ---------------------------------------------------------------------------

void ReplayBuffer::alloc_storage() {
    int C = cfg_.capacity;
    s_observations_   = new float  [static_cast<size_t>(C) * obs_n_]();
    s_actions_        = new int32_t[static_cast<size_t>(C) * act_n_]();
    s_policy_targets_ = new float  [static_cast<size_t>(C) * pol_n_]();
    s_value_targets_  = new float  [static_cast<size_t>(C) * val_n_]();
    s_reward_targets_ = new float  [static_cast<size_t>(C) * rew_n_]();
    s_agent_orders_   = new int32_t[static_cast<size_t>(C) * ord_n_]();
    s_priorities_     = new float  [static_cast<size_t>(C)]();
}

void ReplayBuffer::free_storage() noexcept {
    delete[] s_observations_;
    delete[] s_actions_;
    delete[] s_policy_targets_;
    delete[] s_value_targets_;
    delete[] s_reward_targets_;
    delete[] s_agent_orders_;
    delete[] s_priorities_;
    s_observations_ = s_policy_targets_ = s_value_targets_ = s_reward_targets_ = nullptr;
    s_actions_ = nullptr;
    s_agent_orders_ = nullptr;
    s_priorities_ = nullptr;
}

// ---------------------------------------------------------------------------
// Pinned output buffer allocation
// ---------------------------------------------------------------------------

void ReplayBuffer::alloc_pinned(PinnedBatch& pb, int B) {
    free_pinned(pb);

    auto [obs_ptr,  obs_pin]  = pinned_malloc(sizeof(float)   * B * obs_n_);
    auto [act_ptr,  act_pin]  = pinned_malloc(sizeof(int32_t) * B * act_n_);
    auto [pol_ptr,  pol_pin]  = pinned_malloc(sizeof(float)   * B * pol_n_);
    auto [val_ptr,  val_pin]  = pinned_malloc(sizeof(float)   * B * val_n_);
    auto [rew_ptr,  rew_pin]  = pinned_malloc(sizeof(float)   * B * rew_n_);
    auto [ord_ptr,  ord_pin]  = pinned_malloc(sizeof(int32_t) * B * ord_n_);
    auto [wgt_ptr,  wgt_pin]  = pinned_malloc(sizeof(float)   * B);
    auto [idx_ptr,  idx_pin]  = pinned_malloc(sizeof(int64_t) * B);

    pb.observations   = static_cast<float*>(obs_ptr);
    pb.actions        = static_cast<int32_t*>(act_ptr);
    pb.policy_targets = static_cast<float*>(pol_ptr);
    pb.value_targets  = static_cast<float*>(val_ptr);
    pb.reward_targets = static_cast<float*>(rew_ptr);
    pb.agent_orders   = static_cast<int32_t*>(ord_ptr);
    pb.weights        = static_cast<float*>(wgt_ptr);
    pb.indices        = static_cast<int64_t*>(idx_ptr);
    pb.batch_size     = B;
    // Mark pinned if ALL fields are pinned (conservative).
    pb.is_pinned      = obs_pin && act_pin && pol_pin && val_pin
                     && rew_pin && ord_pin && wgt_pin && idx_pin;
}

void ReplayBuffer::free_pinned(PinnedBatch& pb) noexcept {
    bool p = pb.is_pinned;
    pinned_free(pb.observations,   p);
    pinned_free(pb.actions,        p);
    pinned_free(pb.policy_targets, p);
    pinned_free(pb.value_targets,  p);
    pinned_free(pb.reward_targets, p);
    pinned_free(pb.agent_orders,   p);
    pinned_free(pb.weights,        p);
    pinned_free(pb.indices,        p);
    pb = PinnedBatch{};
}

void ReplayBuffer::ensure_sample_out(int batch_size) {
    if (sample_out_.batch_size != batch_size)
        alloc_pinned(sample_out_, batch_size);
}

void ReplayBuffer::ensure_reanalysis_out(int batch_size) {
    if (reanalysis_out_.batch_size != batch_size)
        alloc_pinned(reanalysis_out_, batch_size);
}

// ---------------------------------------------------------------------------
// PRNG helpers
// ---------------------------------------------------------------------------

float ReplayBuffer::xorshift_uniform() noexcept {
    return ::xorshift_uniform(rng_state_);
}

uint64_t ReplayBuffer::xorshift_uint64() noexcept {
    rng_state_ ^= rng_state_ << 13;
    rng_state_ ^= rng_state_ >> 7;
    rng_state_ ^= rng_state_ << 17;
    return rng_state_;
}

// ---------------------------------------------------------------------------
// add()
// ---------------------------------------------------------------------------

void ReplayBuffer::add(
    const float*   observation,
    const int32_t* actions,
    const float*   policy_target,
    const float*   value_target,
    const float*   reward_target,
    const int32_t* agent_order,
    float          priority)
{
    // Default priority: max existing, or 1.0 on first add.
    if (priority <= 0.0f) {
        priority = sum_tree_.max_priority();
        if (priority <= 0.0f) priority = 1.0f;
    }

    // Claim a slot in the ring buffer.
    int64_t slot_raw = write_ptr_.fetch_add(1, std::memory_order_relaxed);
    int slot = static_cast<int>(slot_raw % cfg_.capacity);

    // Copy each field into storage.
    std::memcpy(s_observations_   + static_cast<size_t>(slot) * obs_n_, observation,  obs_n_ * sizeof(float));
    std::memcpy(s_actions_        + static_cast<size_t>(slot) * act_n_, actions,      act_n_ * sizeof(int32_t));
    std::memcpy(s_policy_targets_ + static_cast<size_t>(slot) * pol_n_, policy_target,pol_n_ * sizeof(float));
    std::memcpy(s_value_targets_  + static_cast<size_t>(slot) * val_n_, value_target, val_n_ * sizeof(float));
    std::memcpy(s_reward_targets_ + static_cast<size_t>(slot) * rew_n_, reward_target,rew_n_ * sizeof(float));
    std::memcpy(s_agent_orders_   + static_cast<size_t>(slot) * ord_n_, agent_order,  ord_n_ * sizeof(int32_t));
    s_priorities_[slot] = priority;

    // Saturating size increment.
    int64_t old_sz = size_.load(std::memory_order_relaxed);
    if (old_sz < static_cast<int64_t>(cfg_.capacity))
        size_.fetch_add(1, std::memory_order_relaxed);

    sum_tree_.update(slot, priority);
}

// ---------------------------------------------------------------------------
// gather helpers
// ---------------------------------------------------------------------------

void ReplayBuffer::gather(PinnedBatch& pb, const int64_t* indices, int n) const noexcept {
    for (int i = 0; i < n; ++i) {
        int idx = static_cast<int>(indices[i]);
        std::memcpy(pb.observations   + static_cast<size_t>(i) * obs_n_,
                    s_observations_   + static_cast<size_t>(idx) * obs_n_,
                    obs_n_ * sizeof(float));
        std::memcpy(pb.actions        + static_cast<size_t>(i) * act_n_,
                    s_actions_        + static_cast<size_t>(idx) * act_n_,
                    act_n_ * sizeof(int32_t));
        std::memcpy(pb.policy_targets + static_cast<size_t>(i) * pol_n_,
                    s_policy_targets_ + static_cast<size_t>(idx) * pol_n_,
                    pol_n_ * sizeof(float));
        std::memcpy(pb.value_targets  + static_cast<size_t>(i) * val_n_,
                    s_value_targets_  + static_cast<size_t>(idx) * val_n_,
                    val_n_ * sizeof(float));
        std::memcpy(pb.reward_targets + static_cast<size_t>(i) * rew_n_,
                    s_reward_targets_ + static_cast<size_t>(idx) * rew_n_,
                    rew_n_ * sizeof(float));
        std::memcpy(pb.agent_orders   + static_cast<size_t>(i) * ord_n_,
                    s_agent_orders_   + static_cast<size_t>(idx) * ord_n_,
                    ord_n_ * sizeof(int32_t));
        pb.indices[i] = idx;
    }
}

void ReplayBuffer::gather_reanalysis(PinnedBatch& pb, const int64_t* indices, int n) const noexcept {
    for (int i = 0; i < n; ++i) {
        int idx = static_cast<int>(indices[i]);
        std::memcpy(pb.observations + static_cast<size_t>(i) * obs_n_,
                    s_observations_ + static_cast<size_t>(idx) * obs_n_,
                    obs_n_ * sizeof(float));
        std::memcpy(pb.agent_orders + static_cast<size_t>(i) * ord_n_,
                    s_agent_orders_ + static_cast<size_t>(idx) * ord_n_,
                    ord_n_ * sizeof(int32_t));
        pb.indices[i] = idx;
    }
}

// ---------------------------------------------------------------------------
// sample()
// ---------------------------------------------------------------------------

bool ReplayBuffer::sample(int batch_size) {
    int n = size();
    if (n == 0) return false;

    ensure_sample_out(batch_size);

    float beta = current_beta();
    frame_count_.fetch_add(1, std::memory_order_relaxed);

    // Temporary index/prob arrays on the stack for small batches, heap otherwise.
    std::vector<int>   tmp_idx(batch_size);
    std::vector<float> tmp_probs(batch_size);

    sum_tree_.sample(batch_size, n, tmp_idx.data(), tmp_probs.data(), rng_state_);

    // Compute IS weights: w_i = (1 / (n * prob_i))^beta, normalize by max.
    float max_w = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        float w = std::pow(1.0f / (static_cast<float>(n) * tmp_probs[i]), beta);
        sample_out_.weights[i] = w;
        if (w > max_w) max_w = w;
    }
    float inv_max = (max_w > 0.0f) ? 1.0f / max_w : 1.0f;
    for (int i = 0; i < batch_size; ++i)
        sample_out_.weights[i] *= inv_max;

    // Convert int indices → int64 for pybind11 / numpy compatibility.
    std::vector<int64_t> idx64(batch_size);
    for (int i = 0; i < batch_size; ++i) idx64[i] = tmp_idx[i];

    gather(sample_out_, idx64.data(), batch_size);
    return true;
}

// ---------------------------------------------------------------------------
// sample_for_reanalysis() — Vitter's Algorithm R (uniform without replacement)
// ---------------------------------------------------------------------------

bool ReplayBuffer::sample_for_reanalysis(int batch_size) {
    int n = size();
    if (n == 0) return false;

    int B = std::min(batch_size, n);
    ensure_reanalysis_out(B);

    // Reservoir sampling: O(n) with no extra allocation beyond the output buffer.
    std::vector<int64_t> reservoir(B);
    for (int i = 0; i < B; ++i) reservoir[i] = i;
    for (int i = B; i < n; ++i) {
        int64_t j = static_cast<int64_t>(xorshift_uint64() % static_cast<uint64_t>(i + 1));
        if (j < B) reservoir[j] = i;
    }

    gather_reanalysis(reanalysis_out_, reservoir.data(), B);
    reanalysis_out_.batch_size = B;
    return true;
}

// ---------------------------------------------------------------------------
// update_priorities()
// ---------------------------------------------------------------------------

void ReplayBuffer::update_priorities(const int64_t* indices,
                                     const float*   priorities, int n) {
    for (int i = 0; i < n; ++i) {
        int slot = static_cast<int>(indices[i]);
        float p  = std::max(priorities[i], 1e-8f);
        s_priorities_[slot] = p;
        sum_tree_.update(slot, p);
    }
}

// ---------------------------------------------------------------------------
// update_targets() — in-place write at unroll position 0 only
// ---------------------------------------------------------------------------

void ReplayBuffer::update_targets(const int64_t* indices,
                                  const float*   policy_targets,
                                  const float*   root_values,
                                  int n) {
    const int N = cfg_.num_agents;
    const int A = cfg_.action_space_size;
    // position-0 policy target size = N * A floats
    const int pol0_n = N * A;
    // position-0 value target size = N floats

    for (int i = 0; i < n; ++i) {
        int idx = static_cast<int>(indices[i]);

        // policy_targets[idx, 0, :, :] = policy_targets_in[i, :, :]
        std::memcpy(s_policy_targets_ + static_cast<size_t>(idx) * pol_n_,
                    policy_targets    + static_cast<size_t>(i)   * pol0_n,
                    pol0_n * sizeof(float));

        // value_targets[idx, 0, :] = root_values[i] broadcast over N agents
        float* vt = s_value_targets_ + static_cast<size_t>(idx) * val_n_;
        float  rv = root_values[i];
        for (int j = 0; j < N; ++j) vt[j] = rv;
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int ReplayBuffer::size() const {
    return static_cast<int>(
        std::min(size_.load(std::memory_order_relaxed),
                 static_cast<int64_t>(cfg_.capacity)));
}

float ReplayBuffer::current_beta() const {
    int64_t fc = frame_count_.load(std::memory_order_relaxed);
    float t = static_cast<float>(fc) / static_cast<float>(cfg_.beta_frames);
    return std::min(1.0f, cfg_.beta_start + t * (1.0f - cfg_.beta_start));
}

ReplayBuffer::Stats ReplayBuffer::get_stats() const {
    int n = size();
    Stats s{};
    s.size     = n;
    s.capacity = cfg_.capacity;
    s.fill_pct = 100.0f * static_cast<float>(n) / static_cast<float>(cfg_.capacity);
    s.beta     = current_beta();
    if (n == 0) return s;

    float p_min = s_priorities_[0], p_max = s_priorities_[0], p_sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float p = s_priorities_[i];
        if (p < p_min) p_min = p;
        if (p > p_max) p_max = p;
        p_sum += p;
    }
    float p_mean = p_sum / static_cast<float>(n);
    float p_var  = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = s_priorities_[i] - p_mean;
        p_var += d * d;
    }
    s.priority_min  = p_min;
    s.priority_max  = p_max;
    s.priority_mean = p_mean;
    s.priority_std  = std::sqrt(p_var / static_cast<float>(n));
    return s;
}
