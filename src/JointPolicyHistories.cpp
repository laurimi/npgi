// JointPolicyHistories.cpp
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

#include "JointPolicyHistories.h"
#include <boost/random/uniform_int_distribution.hpp>
#include "GraphUtilities.hpp"
#include "PolicyGraph.h"
#include "DecPOMDPUtilities.h"
#include "combinations.hpp"

namespace pgi {

std::vector<std::vector<History>> local_histories_at(
    const std::vector<vertex_t>& qv, const JointPolicy& jp) {
  std::vector<std::vector<History>> h(jp.num_agents());
  for (std::size_t i = 0; i < jp.num_agents(); ++i) {
    const PolicyGraph& local = jp.local_policy(i);
    vertex_t start = find_root(local);
    all_paths(start, qv[i], local, PathReporter(&h[i], local));

    if (h[i].empty()) return std::vector<std::vector<History>>();
  }
  return h;
}

std::pair<History, bool> sample_joint_history_from_policy(
    JointPolicy::joint_vertex_t q, const JointPolicy& jp,
    const JointActionSpace& jas, const JointObservationSpace& jos, PRNG& rng) {
  const std::vector<vertex_t> qv = jp.to_local(q);
  const std::vector<std::vector<History>> local_histories =
      local_histories_at(qv, jp);

  History h;
  if (local_histories.empty()) return std::make_pair(h, false);

  // reservoir sampling
  int i = 0;
  for (auto c : make_combinations(local_histories)) {
    ++i;
    const double p_swap = 1.0 / static_cast<double>(i);
    if (is_rnd01_below(p_swap, rng)) {
      h = local_to_joint_history(get_combination(c), jas, jos);
    }
  }
  return std::make_pair(h, true);
}

std::pair<PolicyState, bool> sample_history_and_joint_vertex_from_policy(
    const JointActionSpace& jas, const JointObservationSpace& jos,
    const JointPolicy& jp, unsigned int steps_remaining, PRNG& rng) {
  // reservoir sampling
  int i = 0;
  std::pair<PolicyState, bool> selected(PolicyState(), false);
  for (const auto& q :
       jp.joint_vertices_with_steps_remaining(steps_remaining)) {
    const std::vector<vertex_t> qv = jp.to_local(q);
    const std::vector<std::vector<History>> local_histories = local_histories_at(qv, jp);
    if (local_histories.empty())
      continue;

    for (auto h : make_combinations(local_histories)) {
      ++i;
      const double p_swap = 1.0 / static_cast<double>(i);
      if (is_rnd01_below(p_swap,
                       rng))
      {
        const History history = local_to_joint_history(
            get_combination(h), jas, jos);
        selected = std::make_pair(PolicyState{history, q}, true);
      }
    }
  }
  return selected;
}

std::pair<PolicyState, bool> sample_random_history_and_joint_vertex(
    const DecPOMDPDiscrete& d, const belief_t& b0,
    const JointPolicy& jp, std::size_t idx_local, vertex_t qlocal, PRNG& rng) {
  const PolicyGraph& local = jp.local_policy(idx_local);
  unsigned int hist_length = jp.max_steps() - steps_remaining(qlocal, local);
  auto decpomdphistory = sample_random_joint_history(d, b0, rng, hist_length);

  std::vector<JointPolicy::joint_vertex_t> qj =
      jp.joint_vertices_with_agent_at(idx_local, qlocal);
  boost::random::uniform_int_distribution<std::size_t> vertex_dist(
      0, qj.size() - 1);

  return std::make_pair(PolicyState{decpomdphistory.h_, qj[rng(vertex_dist)]},
                        true);
}

}  // namespace pgi
