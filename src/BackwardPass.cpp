// BackwardPass.cpp
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

#include "BackwardPass.h"
#include <iostream>
#include "JointPolicyHistories.h"

namespace pgi {
namespace backpass {
ImprovementResult improve(JointPolicy jp, const DecPOMDPDiscrete& decpomdp,
                          HistoryCache& cache, PRNG& rng,
                          const BackPassProperties& props) {
  double policy_value = 0.0;
  for (std::size_t steps = jp.min_steps(); steps <= jp.max_steps(); ++steps) {
    for (std::size_t agent = 0; agent < jp.num_agents(); ++agent) {
      PolicyGraph& local = jp.local_policy(agent);
      std::vector<vertex_t> already_improved;
      for (const auto qlocal : boost::make_iterator_range(
               vertices_with_steps_remaining(steps, local))) {
        value_map_t vm = estimate_local_policy_values(
            jp, agent, qlocal, decpomdp, cache, rng, props);

        auto imax = std::max_element(
            vm.begin(), vm.end(), [](const std::pair<std::size_t, stats_t>& a,
                                     const std::pair<std::size_t, stats_t>& b) {
              return boost::accumulators::weighted_sum(a.second) <
                     boost::accumulators::weighted_sum(b.second);
            });
        policy_value = boost::accumulators::weighted_sum(imax->second);
        set_local_policy(qlocal, imax->first, local);
        already_improved.push_back(qlocal);
        cache.ensure_size_within_limits();
      }
    }
  }
  return ImprovementResult{jp, policy_value};
}

value_map_t estimate_local_policy_values(
    const JointPolicy& jp, const std::size_t agent_idx,
    const vertex_t agent_vertex, const DecPOMDPDiscrete& decpomdp,
    HistoryCache& cache, PRNG& rng, const BackPassProperties& props) {
  const bool is_root = (agent_vertex == find_root(jp.local_policy(agent_idx)));
  if (!is_root && is_rnd01_below(props.prob_heuristic_improvement, rng)) {
    return local_node_value::heuristic_estimate(
        decpomdp, cache, jp, agent_idx, agent_vertex, rng,
        is_rnd01_below(props.prob_random_history_in_heuristic_improvement,
                       rng));
  }

  value_map_t local_policy_values;
  if (is_root || !props.use_lower_bound) {
    local_policy_values = local_node_value::exact(
        decpomdp, cache, jp, agent_idx, agent_vertex,
        jp.joint_vertices_with_agent_at(agent_idx, agent_vertex));
  } else {
    local_policy_values = local_node_value::lower_bound(
        decpomdp, cache, jp, agent_idx, agent_vertex,
        jp.joint_vertices_with_agent_at(agent_idx, agent_vertex));
  }

  if (!local_policy_values.empty())
    return local_policy_values;
  else {
    return local_node_value::heuristic_estimate(
        decpomdp, cache, jp, agent_idx, agent_vertex, rng,
        is_rnd01_below(props.prob_random_history_in_heuristic_improvement,
                       rng));
  }
}

}  // namespace backpass
}  // namespace pgi
