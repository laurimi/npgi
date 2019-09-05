// MADPWrapper.h
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

#ifndef MADPWRAPPER_H
#define MADPWRAPPER_H
#include <string>
#include <vector>
#include "madp/DecPOMDPDiscrete.h"

namespace pgi {
namespace madpwrapper {
class MADPDecPOMDPDiscrete {
 public:
  MADPDecPOMDPDiscrete(const std::string& dpomdp_filename);
  std::size_t num_agents() const;
  std::size_t num_states() const;
  std::size_t num_joint_actions() const;
  std::size_t num_joint_observations() const;
  std::size_t num_actions(std::size_t agent) const;
  std::size_t num_observations(std::size_t agent) const;
  std::string state_name(std::size_t state_index) const;
  std::string action_name(std::size_t agent, std::size_t action_index) const;
  std::string observation_name(std::size_t agent,
                               std::size_t observation_index) const;

  double initial_belief_at(std::size_t state) const;
  double reward(std::size_t state, std::size_t j_act) const;
  double observation_probability(std::size_t j_obs, std::size_t state,
                                 std::size_t j_act) const;
  double transition_probability(std::size_t new_state, std::size_t state,
                                std::size_t j_act) const;

 private:
  // :: refers to global namespace to get access to MADP toolbox
  std::unique_ptr<::DecPOMDPDiscrete> dpomdp_;
};
}  // namespace madpwrapper
}  // namespace pgi

#endif  // MADPWRAPPER_H
