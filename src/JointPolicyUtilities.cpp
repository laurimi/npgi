// JointPolicyUtilities.cpp
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

#include "JointPolicyUtilities.h"
#include <algorithm>
#include "GraphUtilities.hpp"
namespace pgi {
unsigned int find_max_steps(const std::vector<PolicyGraph>& locals) {
  auto imax = std::max_element(locals.begin(), locals.end(),
                               [](const PolicyGraph& a, const PolicyGraph& b) {
                                 return (a[boost::graph_bundle].num_steps_ <
                                         b[boost::graph_bundle].num_steps_);
                               });

  return (*imax)[boost::graph_bundle].num_steps_;
}

std::vector<std::vector<vertex_t> > vertices_with_n_steps(
    unsigned int n, const std::vector<PolicyGraph>& locals) {
  std::vector<std::vector<vertex_t> > v;
  for (auto& l : locals) v.emplace_back(vertices_with_n_steps(n, l));
  return v;
}

std::vector<vertex_t> find_root(const std::vector<PolicyGraph>& locals) {
  std::vector<vertex_t> r;
  for (auto& l : locals) r.push_back(find_root(l));
  return r;
}

std::vector<std::size_t> local_actions(const std::vector<vertex_t>& q,
                                       const std::vector<PolicyGraph>& locals) {
  std::vector<std::size_t> act(q.size());
  for (std::size_t i = 0; i < q.size(); ++i)
    act[i] = locals[i][q[i]];
  return act;
}

std::size_t best_greedy_joint_action(belief_t belief,
                                     const DecPOMDPDiscrete& decpomdp,
                                     bool is_final_step) {
  std::size_t greedy_act = 0;
  double max_r = -std::numeric_limits<double>::max();
  for (std::size_t j_act = 0; j_act < decpomdp.joint_action_space().num_joint_indices(); ++j_act) {
    double r = reward(belief, decpomdp.reward_model(), j_act);
    if (is_final_step) {
      r += final_reward(predicted(belief, decpomdp.transition_model(), j_act), decpomdp.reward_model());
    }
    if (r > max_r) {
      max_r = r;
      greedy_act = j_act;
    }
  }
  return greedy_act;
}

void set_open_loop_greedy(std::vector<PolicyGraph>& locals,
                          belief_t belief,
                          const DecPOMDPDiscrete& decpomdp) {
  auto q = find_root(locals);

  for (unsigned int h = find_max_steps(locals); h >= 1; --h) {
    bool final_action = (h == 1);
    std::size_t j_greedy_act =
        best_greedy_joint_action(belief, decpomdp, final_action);
    for (std::size_t i = 0; i < q.size(); ++i) {
      locals[i][q[i]] = decpomdp.joint_action_space().local_index(j_greedy_act, i);
    }

    if (final_action) break;

    belief = predicted(belief, decpomdp.transition_model(), j_greedy_act);

    // redirect to new target nodes
    std::vector<vertex_t> q_next(q);
    for (std::size_t i = 0; i < q.size(); ++i) {
      auto q_at_h = possible_successors(q[i], locals[i]);
      q_next[i] = *q_at_h.first;
      redirect_out_edges(q[i], q_next[i], locals[i]);
    }
    std::swap(q, q_next);
  }
}

void set_blind(std::vector<PolicyGraph>& locals, std::size_t joint_action,
               const JointActionSpace& jas) {
  for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent) {
    PolicyGraph& l = locals[agent];
    for (auto v : boost::make_iterator_range(boost::vertices(l))) {
      l[v] = jas.local_index(joint_action, agent);
    }
  }
}

void set_random(std::vector<PolicyGraph>& locals, PRNG& rng) {
  for (auto& lp : locals) {
    randomize(lp, rng);
  }
}

}  // namespace pgi
