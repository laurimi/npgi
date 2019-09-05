// JointPolicyHistory.h
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

#ifndef JOINTPOLICYHISTORY_H
#define JOINTPOLICYHISTORY_H
#include <map>
#include <tuple>
#include "DecPOMDPDiscrete.h"
#include "LocalPolicyHistory.h"
namespace pgi {
class JointPolicyHistory {
 public:
  JointPolicyHistory(const std::vector<std::string>& local_policy_filenames);

  unsigned int min_steps() const { return 1; }
  unsigned int max_steps() const { return max_steps_; }
  std::pair<std::size_t, bool> next_action_index(
      const History& h, const DecPOMDPDiscrete& decpomdp) const;
  std::pair<std::size_t, bool> next_action_index(
      const std::vector<History>& local_histories,
      const DecPOMDPDiscrete& decpomdp) const;

 private:
  std::map<std::vector<unsigned int>, std::size_t>
      observation_history_to_action_;
  unsigned int max_steps_;
  std::vector<LocalPolicyHistory> local_;
};
}  // namespace pgi

#endif  // JOINTPOLICYHISTORY_H
