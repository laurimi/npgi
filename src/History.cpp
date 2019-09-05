// History.cpp
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

#include "History.h"
namespace pgi {

std::size_t joint_action_at_step(const std::vector<History>& locals,
                                 const JointActionSpace& jas,
                                 std::size_t step) {
  std::vector<std::size_t> local_actions;
  std::transform(
      locals.begin(), locals.end(), std::back_inserter(local_actions),
      [step](const History& hlocal) { return hlocal[step].action_index_; });
  return jas.joint_index(local_actions);
}

std::size_t joint_observation_at_step(const std::vector<History>& locals,
                                      const JointObservationSpace& jos,
                                      std::size_t step) {
  std::vector<std::size_t> local_observations;
  std::transform(locals.begin(), locals.end(),
                 std::back_inserter(local_observations),
                 [step](const History& hlocal) {
                   return hlocal[step].observation_index_;
                 });
  return jos.joint_index(local_observations);
}

History local_to_joint_history(const std::vector<History>& locals,
                               const JointActionSpace& jas,
                               const JointObservationSpace& jos) {
  History joint;
  joint.reserve(locals[0].size());
  for (std::size_t step = 0; step < locals[0].size(); ++step) {
    joint.emplace_back(
        ActionObservation{joint_action_at_step(locals, jas, step),
                          joint_observation_at_step(locals, jos, step)});
  }
  return joint;
}

ActionObservation joint_to_local(const ActionObservation& joint,
                                 const JointActionSpace& jas,
                                 const JointObservationSpace& jos,
                                 std::size_t agent_idx) {
  return ActionObservation{
      jas.local_index(joint.action_index_, agent_idx),
      jos.local_index(joint.observation_index_, agent_idx)};
}

std::vector<History> joint_to_local_history(
    const History& joint, const DecPOMDPDiscrete& decpomdp) {
  std::vector<History> local(decpomdp.num_agents());
  for (const auto& jao : joint) {
    for (std::size_t agent = 0; agent < decpomdp.num_agents(); ++agent)
      local[agent].push_back(joint_to_local(jao, decpomdp.joint_action_space(),
                                            decpomdp.joint_observation_space(),
                                            agent));
  }
  return local;
}

}  // namespace pgi
