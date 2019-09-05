// BeliefUtilities.cpp
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

#include "BeliefUtilities.h"
#include "HistoryCacheUtils.h"
#include "JointPolicyHistories.h"
#include "combinations.hpp"

namespace pgi {
std::pair<belief_t, double> joint_belief_with_probability(
    JointPolicy::joint_vertex_t q, const JointPolicy& jp,
    const DecPOMDPDiscrete& decpomdp, HistoryCache& c) {
  typedef std::vector<History> HistoryVector;

  const std::vector<vertex_t> qv = jp.to_local(q);
  const std::vector<HistoryVector> local_histories = local_histories_at(qv, jp);

  if (local_histories.empty())
    return std::make_pair(belief_t(Eigen::VectorXd(0)), 0.0);

  Eigen::Matrix<double, Eigen::Dynamic, 1> b =
      Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
          decpomdp.state_space().size());
  double prob = 0.0;

  const bool final_reward_at_end = jp.is_terminal(q);
  for (auto h : make_combinations(local_histories)) {
    HistoryData hd = get_data_or_insert_missing(
        local_to_joint_history(get_combination(h),
                               decpomdp.joint_action_space(),
                               decpomdp.joint_observation_space()),
        c, decpomdp, final_reward_at_end);
    if (is_reachable(hd)) {
      prob += hd.probability_;
      b += hd.probability_ * as_vector(hd.belief_);
    }

    c.ensure_size_within_limits();
  }
  if (!is_almost_zero(prob)) b /= prob;

  return std::make_pair(b, prob);
}

std::pair<belief_t, double> expected_belief_with_probability(
    const std::vector<History>& histories, HistoryCache& c,
    const DecPOMDPDiscrete& decpomdp, bool final_reward_at_end) {
  if (histories.empty())
    return std::make_pair(belief_t(Eigen::VectorXd(0)), 0.0);

  Eigen::Matrix<double, Eigen::Dynamic, 1> b =
      Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(
          decpomdp.state_space().size());
  double prob = 0.0;
  for (auto& h : histories) {
    HistoryData hd =
        get_data_or_insert_missing(h, c, decpomdp, final_reward_at_end);
    if (is_reachable(hd)) {
      prob += hd.probability_;

      (b += hd.probability_ * as_vector(hd.belief_)).eval();
    }

    c.ensure_size_within_limits();
  }
  if (!is_almost_zero(prob)) b /= prob;

  return std::make_pair(b, prob);
}
}  // namespace pgi
