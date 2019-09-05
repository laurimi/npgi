// ValueUtilities.cpp
// Copyright 2019 Mikko Lauri
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ValueUtilities.h"
#include "ValueFunction.h"
#include "BeliefUtilities.h"
#include "JointPolicyHistories.h"
#include "combinations.hpp"
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int.hpp>

namespace pgi {

std::size_t best_value_index(const value_map_t& vm) {
  auto it = std::max_element(
      vm.begin(), vm.end(), [](const std::pair<std::size_t, stats_t>& a,
                               const std::pair<std::size_t, stats_t>& b) {
        return boost::accumulators::weighted_sum(a.second) <
               boost::accumulators::weighted_sum(b.second);
      });
  return it->first;
}

namespace local_node_value {
// lower bound for the values of local policies of node qlocal in local policy
// idx_local, evaluated by taking the expectation under expected beliefs of
// joint nodes in qj.
value_map_t lower_bound(const DecPOMDPDiscrete& d, HistoryCache& hc,
                        JointPolicy jp, std::size_t idx_local, vertex_t qlocal,
                        const std::vector<JointPolicy::joint_vertex_t>& qj) {
  value_map_t vm;
  PolicyGraph& local = jp.local_policy(idx_local);
  for (const auto& q : qj) {
    auto jb = joint_belief_with_probability(q, jp,d, hc);

    if (is_almost_zero(jb.second)) continue;
    for (std::size_t a = 0; a < local[boost::graph_bundle].num_actions_; ++a) {
      HistoryCache hcache(HistoryData{1.0, 0.0, 0.0, jb.first});

      local[qlocal] = a;
      for (std::size_t e = 0; e < num_out_edge_configurations(qlocal, local);
           ++e) {
        set_out_edge_configuration(qlocal, e, local);
        double value = pgi::value(jp, q, History(), d, hcache);
        const std::size_t i = get_local_policy(qlocal, a, e, local);
        vm[i](value, boost::accumulators::weight = jb.second);
      }
    }
  }
  return vm;
}

value_map_t exact(const DecPOMDPDiscrete& d,
                  HistoryCache& hc, JointPolicy jp,
                  std::size_t idx_local, vertex_t qlocal,
                  const std::vector<JointPolicy::joint_vertex_t>& qj) {
  value_map_t vm;
  PolicyGraph& local = jp.local_policy(idx_local);
  for (std::size_t a = 0; a < local[boost::graph_bundle].num_actions_; ++a) {
    local[qlocal] = a;
    for (std::size_t e = 0; e < num_out_edge_configurations(qlocal, local);
         ++e) {
      set_out_edge_configuration(qlocal, e, local);
      const std::size_t i = get_local_policy(qlocal, a, e, local);

      for (const auto& q : qj) {
        const std::vector<vertex_t> qv = jp.to_local(q);
        const std::vector<std::vector<History>> local_histories = local_histories_at(qv, jp);

        if (local_histories.empty())
          continue;

        for (auto h : make_combinations(local_histories)) {
          const History history = local_to_joint_history(get_combination(h),
                                     d.joint_action_space(),
                                     d.joint_observation_space());
          const HistoryData& hd = get_data_or_insert_missing(history,
              hc, d, jp.is_terminal(q));
          if (!is_reachable(hd))
            continue;

          double value = pgi::value(jp, q, history, d, hc);
          vm[i](value, boost::accumulators::weight = hd.probability_);
          hc.ensure_size_within_limits();
        }
      }
    }
  }
  return vm;
}

value_map_t heuristic_estimate(const DecPOMDPDiscrete& d,
                               HistoryCache& hc,
                               const JointPolicy& jp, std::size_t idx_local,
                               vertex_t qlocal, PRNG& rng,
                               bool use_random_history) {
  const PolicyGraph& local = jp.local_policy(idx_local);
  std::pair<PolicyState, bool> ps(PolicyState{}, false);
  if (!use_random_history) {
    ps = sample_history_and_joint_vertex_from_policy(
        d.joint_action_space(), d.joint_observation_space(), jp,
        steps_remaining(qlocal, local), rng);
  }
  if (!ps.second) {
    const HistoryData& rootdata = get_data_or_insert_missing(
        History(), hc, d, jp.is_terminal(ps.first.joint_vertex_));
    ps = sample_random_history_and_joint_vertex(d, rootdata.belief_, jp,
                                                idx_local, qlocal, rng);
  }

  return policy_state_value(d, jp, ps.first, idx_local, qlocal, hc);
}

value_map_t policy_state_value(const DecPOMDPDiscrete& d,
                               JointPolicy jp, const PolicyState& ps,
                               std::size_t idx_local, vertex_t qlocal,
                               HistoryCache& hc) {
  value_map_t vm;
  PolicyGraph& local = jp.local_policy(idx_local);
  for (std::size_t a = 0; a < local[boost::graph_bundle].num_actions_; ++a) {
    local[qlocal] = a;
    for (std::size_t e = 0; e < num_out_edge_configurations(qlocal, local);
         ++e) {
      set_out_edge_configuration(qlocal, e, local);
      const std::size_t i = get_local_policy(qlocal, a, e, local);
      double value = pgi::value(jp, ps.joint_vertex_, ps.history_, d, hc);
      vm[i](value, boost::accumulators::weight = 1.0);
    }
  }
  return vm;
}

}  // namespace local_node_value
}  // namespace pgi
