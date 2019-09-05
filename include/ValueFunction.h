// ValueFunction.h
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

#ifndef VALUEFUNCTION_H
#define VALUEFUNCTION_H
#include "DecPOMDPDiscrete.h"
#include "History.h"
#include "HistoryCache.hpp"
#include "HistoryCacheUtils.h"
#include "JointPolicyHistory.h"
#include "PolicyGraphTraversal.h"

namespace pgi {

double value(const JointPolicy& jp, JointPolicy::joint_vertex_t qstart,
             const History& h, const DecPOMDPDiscrete& decpomdp,
             HistoryCache& hc);

void value_helper(const DecPOMDPDiscrete&, JointPolicyGraphTraversal& tv,
                  HistoryCache::traversal_t& tree_traversal, HistoryCache& hc,
                  double& value);

class ValueFunction {
 public:
  ValueFunction(std::shared_ptr<HistoryCache> bc) : bc_(bc) {}

  double expected_value(const DecPOMDPDiscrete& decpomdp,
                        const JointPolicyHistory& p) const {
    double value = 0.0;
    History h;
    expected_value_helper(decpomdp, p, h, value);
    return value;
  }

 private:
  void expected_value_helper(const DecPOMDPDiscrete& decpomdp,
                             const JointPolicyHistory& p, History& h,
                             double& V) const {
    auto ap = p.next_action_index(h, decpomdp);
    for (std::size_t j_obs = 0;
         j_obs < decpomdp.joint_observation_space().num_joint_indices();
         ++j_obs) {
      h.emplace_back(ActionObservation{ap.first, j_obs});
      const bool is_final_step = (h.size() == p.max_steps());
      HistoryData new_hist_data =
          get_data_or_insert_missing(h, *bc_, decpomdp, is_final_step);
      if (is_reachable(new_hist_data)) {
        if (is_final_step) {
          V += new_hist_data.probability_ *
               new_hist_data.sum_of_expected_rewards_;
        } else {
          expected_value_helper(decpomdp, p, h, V);
        }
      }
      h.pop_back();
    }
  }
  std::shared_ptr<HistoryCache> bc_;
};
}  // namespace pgi
#endif  // VALUEFUNCTION_H
