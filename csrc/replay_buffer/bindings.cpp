// csrc/replay_buffer/bindings.cpp
//
// pybind11 bindings for ReplayBuffer.
//
// Zero-copy return strategy: numpy arrays returned by sample() and
// sample_for_reanalysis() wrap the underlying pinned C++ buffers directly.
// Each returned array holds `self_obj` (the Python ReplayBuffer proxy) as its
// numpy base, keeping the C++ object (and its pinned memory) alive for as long
// as any array is in scope on the Python side.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "replay_buffer.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: wrap a raw C pointer as a numpy array WITHOUT copying.
// `base` should be the owning Python object (prevents premature deallocation).
// ---------------------------------------------------------------------------
template <typename T>
static py::array_t<T> wrap(T* ptr, std::vector<ssize_t> shape, py::object base) {
    std::vector<ssize_t> strides(shape.size());
    ssize_t s = sizeof(T);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i];
    }
    return py::array_t<T>(shape, strides, ptr, base);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_replay_buffer_cpp, m) {
    m.doc() = "C++ lock-free prioritized replay buffer with optional pinned memory.";

    // ------------------------------------------------------------------
    // ReplayBufferConfig
    // ------------------------------------------------------------------
    py::class_<ReplayBufferConfig>(m, "ReplayBufferConfig")
        .def(py::init<>())
        .def_readwrite("capacity",          &ReplayBufferConfig::capacity)
        .def_readwrite("obs_size",          &ReplayBufferConfig::obs_size)
        .def_readwrite("action_space_size", &ReplayBufferConfig::action_space_size)
        .def_readwrite("num_agents",        &ReplayBufferConfig::num_agents)
        .def_readwrite("unroll_steps",      &ReplayBufferConfig::unroll_steps)
        .def_readwrite("alpha",             &ReplayBufferConfig::alpha)
        .def_readwrite("beta_start",        &ReplayBufferConfig::beta_start)
        .def_readwrite("beta_frames",       &ReplayBufferConfig::beta_frames);

    // ------------------------------------------------------------------
    // ReplayBuffer
    // ------------------------------------------------------------------
    py::class_<ReplayBuffer>(m, "ReplayBuffer")
        .def(py::init<const ReplayBufferConfig&>(), py::arg("config"))

        // ---- add() ---------------------------------------------------
        // Accepts six numpy arrays (C-contiguous, correct dtypes) and a float
        // priority.  Priority <= 0 → use max existing priority.
        .def("add",
            [](ReplayBuffer& self,
               py::array_t<float,   py::array::c_style | py::array::forcecast> obs,
               py::array_t<int32_t, py::array::c_style | py::array::forcecast> actions,
               py::array_t<float,   py::array::c_style | py::array::forcecast> policy_target,
               py::array_t<float,   py::array::c_style | py::array::forcecast> value_target,
               py::array_t<float,   py::array::c_style | py::array::forcecast> reward_target,
               py::array_t<int32_t, py::array::c_style | py::array::forcecast> agent_order,
               float priority)
            {
                self.add(obs.data(), actions.data(), policy_target.data(),
                         value_target.data(), reward_target.data(),
                         agent_order.data(), priority);
            },
            py::arg("observation"), py::arg("actions"), py::arg("policy_target"),
            py::arg("value_target"), py::arg("reward_target"),
            py::arg("agent_order"), py::arg("priority"))

        // ---- sample() ------------------------------------------------
        // Returns (fields_dict, weights, indices) or None if buffer empty.
        // All arrays are zero-copy views of the pinned output buffers.
        .def("sample",
            [](py::object self_obj, int batch_size) -> py::object {
                ReplayBuffer& self = self_obj.cast<ReplayBuffer&>();
                if (!self.sample(batch_size)) return py::none();

                const PinnedBatch& pb  = self.get_sample_out();
                const ReplayBufferConfig& cfg = self.config();
                int B = pb.batch_size;
                int N = cfg.num_agents;
                int A = cfg.action_space_size;
                int U = cfg.unroll_steps;

                py::dict d;
                d["observation"]   = wrap(pb.observations,   {B, N, cfg.obs_size}, self_obj);
                d["actions"]       = wrap(pb.actions,         {B, U, N},            self_obj);
                d["policy_target"] = wrap(pb.policy_targets,  {B, U+1, N, A},       self_obj);
                d["value_target"]  = wrap(pb.value_targets,   {B, U+1, N},          self_obj);
                d["reward_target"] = wrap(pb.reward_targets,  {B, U, N},            self_obj);
                d["agent_order"]   = wrap(pb.agent_orders,    {B, N},               self_obj);

                auto weights = wrap(pb.weights,  {B}, self_obj);
                auto indices = wrap(pb.indices,  {B}, self_obj);
                return py::make_tuple(d, weights, indices);
            },
            py::arg("batch_size"))

        // ---- sample_for_reanalysis() ---------------------------------
        // Returns (indices, observations, agent_orders) or None if empty.
        .def("sample_for_reanalysis",
            [](py::object self_obj, int batch_size) -> py::object {
                ReplayBuffer& self = self_obj.cast<ReplayBuffer&>();
                if (!self.sample_for_reanalysis(batch_size)) return py::none();

                const PinnedBatch& pb  = self.get_reanalysis_out();
                const ReplayBufferConfig& cfg = self.config();
                int B = pb.batch_size;
                int N = cfg.num_agents;

                auto indices = wrap(pb.indices,      {B},             self_obj);
                auto obs     = wrap(pb.observations,  {B, N, cfg.obs_size}, self_obj);
                auto orders  = wrap(pb.agent_orders,  {B, N},         self_obj);
                return py::make_tuple(indices, obs, orders);
            },
            py::arg("batch_size"))

        // ---- update_priorities() -------------------------------------
        .def("update_priorities",
            [](ReplayBuffer& self,
               py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices,
               py::array_t<float,   py::array::c_style | py::array::forcecast> priorities)
            {
                if (indices.size() != priorities.size())
                    throw std::invalid_argument("indices and priorities must have the same length");
                self.update_priorities(indices.data(), priorities.data(),
                                       static_cast<int>(indices.size()));
            },
            py::arg("indices"), py::arg("priorities"))

        // ---- update_targets() ----------------------------------------
        .def("update_targets",
            [](ReplayBuffer& self,
               py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices,
               py::array_t<float,   py::array::c_style | py::array::forcecast> policy_targets,
               py::array_t<float,   py::array::c_style | py::array::forcecast> root_values)
            {
                int n = static_cast<int>(indices.size());
                self.update_targets(indices.data(), policy_targets.data(),
                                    root_values.data(), n);
            },
            py::arg("indices"), py::arg("policy_targets"), py::arg("root_values"))

        // ---- Accessors -----------------------------------------------
        .def("__len__",    &ReplayBuffer::size)
        .def("size",       &ReplayBuffer::size)
        .def("capacity",   &ReplayBuffer::capacity)
        .def("is_pinned",  [](const ReplayBuffer& self) {
            return self.get_sample_out().is_pinned;
        })

        // ---- get_stats() --------------------------------------------
        .def("get_stats",
            [](const ReplayBuffer& self) -> py::dict {
                auto s = self.get_stats();
                py::dict d;
                d["size"]          = s.size;
                d["capacity"]      = s.capacity;
                d["fill_pct"]      = s.fill_pct;
                d["priority_min"]  = s.priority_min;
                d["priority_max"]  = s.priority_max;
                d["priority_mean"] = s.priority_mean;
                d["priority_std"]  = s.priority_std;
                d["beta"]          = s.beta;
                return d;
            });
}
