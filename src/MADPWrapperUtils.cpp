// MADPWrapperUtils.cpp
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

#include "MADPWrapperUtils.h"

namespace pgi {
belief_t make_initial_belief(const madpwrapper::MADPDecPOMDPDiscrete& d,
                             bool sparse) {
  if (sparse)
    return belief_t(detail::make_initial_belief_sparse(d));
  else
    return belief_t(detail::make_initial_belief(d));
}

StateSpace make_state_space(const madpwrapper::MADPDecPOMDPDiscrete& d) {
  std::vector<DiscreteState> s;
  for (std::size_t i = 0; i < d.num_states(); ++i) {
    s.emplace_back(DiscreteState(d.state_name(i)));
  }
  return StateSpace(s);
}

ActionSpace make_action_space(const madpwrapper::MADPDecPOMDPDiscrete& d,
                              std::size_t agent) {
  std::vector<DiscreteAction> a;
  for (std::size_t i = 0; i < d.num_actions(agent); ++i)
    a.emplace_back(DiscreteAction(d.action_name(agent, i)));
  return ActionSpace(a);
}

ObservationSpace make_observation_space(
    const madpwrapper::MADPDecPOMDPDiscrete& d, std::size_t agent) {
  std::vector<DiscreteObservation> o;
  for (std::size_t i = 0; i < d.num_observations(agent); ++i)
    o.emplace_back(DiscreteObservation(d.observation_name(agent, i)));
  return ObservationSpace(o);
}

JointActionSpace make_joint_action_space(
    const madpwrapper::MADPDecPOMDPDiscrete& d) {
  std::vector<ActionSpace> a;
  for (std::size_t agent = 0; agent < d.num_agents(); ++agent)
    a.emplace_back(make_action_space(d, agent));
  return JointActionSpace(a);
}

JointObservationSpace make_joint_observation_space(
    const madpwrapper::MADPDecPOMDPDiscrete& d) {
  std::vector<ObservationSpace> o;
  for (std::size_t agent = 0; agent < d.num_agents(); ++agent)
    o.emplace_back(make_observation_space(d, agent));
  return JointObservationSpace(o);
}

std::unique_ptr<StateTransitionInterface> make_transition_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d, bool sparse) {
  if (sparse)
    return std::make_unique<StateTransitionMatrix<true>>(
        detail::make_transition_matrix_sparse(d));
  else
    return std::make_unique<StateTransitionMatrix<false>>(
        detail::make_transition_matrix(d));
}

std::unique_ptr<ObservationInterface> make_observation_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d, bool sparse) {
  if (sparse)
    return std::make_unique<ObservationMatrix<true>>(
        detail::make_observation_matrix_sparse(d));
  else
    return std::make_unique<ObservationMatrix<false>>(
        detail::make_observation_matrix(d));
}

std::unique_ptr<RewardInterface> make_reward_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d, bool use_final_reward) {
  return std::make_unique<RewardMatrix>(
      detail::make_reward_matrix(d, use_final_reward));
}

namespace detail {
Eigen::VectorXd make_initial_belief(
    const madpwrapper::MADPDecPOMDPDiscrete& d) {
  Eigen::VectorXd b(d.num_states());
  for (std::size_t si = 0; si < d.num_states(); ++si)
    b(si) = d.initial_belief_at(si);
  return b;
}

Eigen::SparseVector<double> make_initial_belief_sparse(
    const madpwrapper::MADPDecPOMDPDiscrete& d) {
  Eigen::SparseVector<double> b(d.num_states());
  for (std::size_t si = 0; si < d.num_states(); ++si) {
    if (!is_almost_zero(d.initial_belief_at(si)))
      b.coeffRef(si) = d.initial_belief_at(si);
  }
  return b;
}

StateTransitionMatrix<false> make_transition_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d) {
  StateTransitionMatrix<false> T(d.num_states(), d.num_joint_actions());
  for (std::size_t j_act = 0; j_act < d.num_joint_actions(); ++j_act)
    for (std::size_t s_next = 0; s_next < d.num_states(); ++s_next)
      for (std::size_t s_old = 0; s_old < d.num_states(); ++s_old)
        T.set(s_next, s_old, j_act,
              d.transition_probability(s_next, s_old, j_act));
  return T;
}

StateTransitionMatrix<true> make_transition_matrix_sparse(
    const madpwrapper::MADPDecPOMDPDiscrete& d) {
  StateTransitionMatrix<true> T(d.num_states(), d.num_joint_actions());
  for (std::size_t j_act = 0; j_act < d.num_joint_actions(); ++j_act)
    for (std::size_t s_next = 0; s_next < d.num_states(); ++s_next)
      for (std::size_t s_old = 0; s_old < d.num_states(); ++s_old)
        if (!is_almost_zero(d.transition_probability(s_next, s_old, j_act)))
          T.set(s_next, s_old, j_act,
                d.transition_probability(s_next, s_old, j_act));
  return T;
}

ObservationMatrix<false> make_observation_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d) {
  ObservationMatrix<false> O(d.num_joint_observations(), d.num_states(),
                             d.num_joint_actions());
  for (std::size_t j_act = 0; j_act < d.num_joint_actions(); ++j_act)
    for (std::size_t si = 0; si < d.num_states(); ++si)
      for (std::size_t j_obs = 0; j_obs < d.num_joint_observations(); ++j_obs)
        O.set(j_obs, si, j_act, d.observation_probability(j_obs, si, j_act));

  return O;
}

ObservationMatrix<true> make_observation_matrix_sparse(
    const madpwrapper::MADPDecPOMDPDiscrete& d) {
  ObservationMatrix<true> O(d.num_joint_observations(), d.num_states(),
                            d.num_joint_actions());
  for (std::size_t j_act = 0; j_act < d.num_joint_actions(); ++j_act)
    for (std::size_t si = 0; si < d.num_states(); ++si)
      for (std::size_t j_obs = 0; j_obs < d.num_joint_observations(); ++j_obs)
        if (!is_almost_zero(d.observation_probability(j_obs, si, j_act)))
          O.set(j_obs, si, j_act, d.observation_probability(j_obs, si, j_act));

  return O;
}

RewardMatrix make_reward_matrix(const madpwrapper::MADPDecPOMDPDiscrete& d,
                                bool use_final_reward) {
  RewardMatrix R(d.num_states(), d.num_joint_actions(), use_final_reward);
  for (std::size_t j_act = 0; j_act < d.num_joint_actions(); ++j_act)
    for (std::size_t si = 0; si < d.num_states(); ++si)
      R.set(si, j_act, d.reward(si, j_act));
  return R;
}

}  // namespace detail
}  // namespace pgi
