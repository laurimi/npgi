// MADPWrapper.cpp
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

#include "MADPWrapper.h"
#include "common.hpp"
#include "madp/MADPParser.h"
#include <memory>
namespace pgi {
namespace madpwrapper {

MADPDecPOMDPDiscrete::MADPDecPOMDPDiscrete(const std::string& dpomdp_filename)
    : dpomdp_(std::make_unique<::DecPOMDPDiscrete>("", "", dpomdp_filename)) {
  MADPParser parser(dpomdp_.get());
}
std::size_t MADPDecPOMDPDiscrete::num_agents() const {
  return dpomdp_->GetNrAgents();
}
std::size_t MADPDecPOMDPDiscrete::num_states() const {
  return dpomdp_->GetNrStates();
}
std::size_t MADPDecPOMDPDiscrete::num_joint_actions() const {
  return dpomdp_->GetNrJointActions();
}
std::size_t MADPDecPOMDPDiscrete::num_joint_observations() const {
  return dpomdp_->GetNrJointObservations();
}
std::size_t MADPDecPOMDPDiscrete::num_actions(std::size_t agent) const {
  return dpomdp_->GetNrActions(agent);
}
std::size_t MADPDecPOMDPDiscrete::num_observations(std::size_t agent) const {
  return dpomdp_->GetNrObservations(agent);
}
std::string MADPDecPOMDPDiscrete::state_name(std::size_t state_index) const {
  return dpomdp_->GetState(state_index)->GetName();
}
std::string MADPDecPOMDPDiscrete::action_name(std::size_t agent,
                                              std::size_t action_index) const {
  return dpomdp_->GetAction(agent, action_index)->GetName();
}
std::string MADPDecPOMDPDiscrete::observation_name(
    std::size_t agent, std::size_t observation_index) const {
  return dpomdp_->GetObservation(agent, observation_index)->GetName();
}
double MADPDecPOMDPDiscrete::initial_belief_at(std::size_t state) const {
  return dpomdp_->GetInitialStateProbability(state);
}
double MADPDecPOMDPDiscrete::reward(std::size_t state,
                                    std::size_t j_act) const {
  return dpomdp_->GetReward(state, j_act);
}
double MADPDecPOMDPDiscrete::observation_probability(std::size_t j_obs,
                                                     std::size_t state,
                                                     std::size_t j_act) const {
  return dpomdp_->GetObservationModelDiscretePtr()->Get(j_act, state, j_obs);
}
double MADPDecPOMDPDiscrete::transition_probability(std::size_t new_state,
                                                    std::size_t state,
                                                    std::size_t j_act) const {
  return dpomdp_->GetTransitionModelDiscretePtr()->Get(state, j_act, new_state);
}
}  // namespace madpwrapper
}  // namespace pgi
