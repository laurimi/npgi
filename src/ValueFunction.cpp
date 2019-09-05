// ValueFunction.cpp
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

#include "ValueFunction.h"
#include "JointPolicyUtilities.h"

namespace pgi {

double value(const JointPolicy& jp, JointPolicy::joint_vertex_t qstart,
             const History& h, const DecPOMDPDiscrete& decpomdp,
             HistoryCache& hc) {
  JointPolicyGraphTraversal tv(qstart, jp);
  const HistoryData& hd = get_data_or_insert_missing(
      h, hc, decpomdp, tv.last_action());  // just ensures this exists
  HistoryCache::traversal_t tree_traversal = hc.get_traversal(h);
  double value = 0.0;
  value_helper(decpomdp, tv, tree_traversal, hc, value);
  return value;
}

void value_helper(const DecPOMDPDiscrete& decpomdp,
                  JointPolicyGraphTraversal& tv,
                  HistoryCache::traversal_t& tree_traversal, HistoryCache& hc,
                  double& value) {
  const JointObservationSpace& jos = decpomdp.joint_observation_space();
  const std::size_t ja = tv.current_action(decpomdp.joint_action_space());
  for (std::size_t jo = 0; jo < jos.num_joint_indices(); ++jo) {
    ActionObservation next_ao{ja, jo};

    auto tp = tree_traversal.can_traverse(next_ao);
    if (tp.second) {
      tree_traversal.traverse(*tp.first);
    } else {
      const HistoryData& parent_data =
          hc.get_data(tree_traversal.current_vertex());
      auto vp = hc.add_vertex(
          tree_traversal.current_vertex(), next_ao,
          get_next(parent_data, decpomdp, next_ao, tv.last_action()));
      tree_traversal.traverse(vp.second);
    }

    const HistoryData& nhd = hc.get_data(tree_traversal.current_vertex());
    if (is_reachable(nhd)) {
      if (tv.last_action())
        value += nhd.probability_ * nhd.sum_of_expected_rewards_;
      else {
        tv.traverse(jo, jos);
        value_helper(decpomdp, tv, tree_traversal, hc, value);
        tv.return_to_previous();
      }
    }
    tree_traversal.return_to_previous();
  }
}

}  // namespace pgi
