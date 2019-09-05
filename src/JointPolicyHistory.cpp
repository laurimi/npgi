// JointPolicyHistory.cpp
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

#include "JointPolicyHistory.h"

namespace pgi {
JointPolicyHistory::JointPolicyHistory(
    const std::vector<std::string>& local_policy_filenames)
    : observation_history_to_action_(), max_steps_(1), local_() {
  for (std::size_t i = 0; i < local_policy_filenames.size(); ++i) {
    local_.push_back(LocalPolicyHistory(local_policy_filenames.at(i)));
    max_steps_ = std::max(local_.back().max_steps(), max_steps_);
  }
}

std::pair<std::size_t, bool> JointPolicyHistory::next_action_index(
    const History& h, const DecPOMDPDiscrete& decpomdp) const {
  return next_action_index(joint_to_local_history(h, decpomdp), decpomdp);
}

std::pair<std::size_t, bool> JointPolicyHistory::next_action_index(
    const std::vector<History>& local_histories,
    const DecPOMDPDiscrete& decpomdp) const {
  const JointActionSpace& jas = decpomdp.joint_action_space();
  std::vector<std::size_t> local_actions(jas.num_local_spaces());
  for (std::size_t i = 0; i < jas.num_local_spaces(); ++i) {
    auto na = local_[i].next_action_index(local_histories[i]);
    if (!na.second) return std::pair<std::size_t, bool>(0, false);

    local_actions[i] = na.first;
  }
  return std::pair<std::size_t, bool>(jas.joint_index(local_actions), true);
}
}
