// HistoryData.cpp
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

#include "HistoryData.h"
#include "common.hpp"

namespace pgi {
bool is_reachable(const HistoryData& d) {
  return !is_almost_zero(d.probability_);
}

HistoryData get_next(const HistoryData& current,
                     const DecPOMDPDiscrete& decpomdp,
                     const ActionObservation& joint, bool add_final_reward) {
  if (!is_reachable(current)) return HistoryData();

  double last_prob = 0.0;
  belief_t next_belief =
      successor(current.belief_, decpomdp.transition_model(),
                decpomdp.observation_model(), joint.action_index_,
                joint.observation_index_, last_prob);

  if (is_almost_zero(last_prob)) return HistoryData();

  double next_expected_reward =
      reward(current.belief_, decpomdp.reward_model(), joint.action_index_);
  if (add_final_reward)
    next_expected_reward += final_reward(next_belief, decpomdp.reward_model());

  return HistoryData{current.probability_ * last_prob, last_prob,
                     next_expected_reward + current.sum_of_expected_rewards_,
                     next_belief};
}

}  // namespace pgi
