// MADPWrapperUtils.h
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

#ifndef MADPWRAPPERUTILS_H
#define MADPWRAPPERUTILS_H
#include "Belief.hpp"
#include "DiscreteAbstractions.hpp"
#include "MADPWrapper.h"
#include "ObservationMatrix.hpp"
#include "RewardMatrix.h"
#include "StateTransitionMatrix.hpp"

namespace pgi {
belief_t make_initial_belief(const madpwrapper::MADPDecPOMDPDiscrete& d,
                             bool sparse);

StateSpace make_state_space(const madpwrapper::MADPDecPOMDPDiscrete& d);
ActionSpace make_action_space(const madpwrapper::MADPDecPOMDPDiscrete& d,
                              std::size_t agent);
ObservationSpace make_observation_space(
    const madpwrapper::MADPDecPOMDPDiscrete& d, std::size_t agent);
JointActionSpace make_joint_action_space(
    const madpwrapper::MADPDecPOMDPDiscrete& d);
JointObservationSpace make_joint_observation_space(
    const madpwrapper::MADPDecPOMDPDiscrete& d);

std::unique_ptr<StateTransitionInterface> make_transition_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d, bool sparse);
std::unique_ptr<ObservationInterface> make_observation_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d, bool sparse);
std::unique_ptr<RewardInterface> make_reward_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d, bool use_final_reward);

namespace detail {
Eigen::VectorXd make_initial_belief(const madpwrapper::MADPDecPOMDPDiscrete& d);
Eigen::SparseVector<double> make_initial_belief_sparse(
    const madpwrapper::MADPDecPOMDPDiscrete& d);
StateTransitionMatrix<false> make_transition_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d);
StateTransitionMatrix<true> make_transition_matrix_sparse(
    const madpwrapper::MADPDecPOMDPDiscrete& d);
ObservationMatrix<false> make_observation_matrix(
    const madpwrapper::MADPDecPOMDPDiscrete& d);
ObservationMatrix<true> make_observation_matrix_sparse(
    const madpwrapper::MADPDecPOMDPDiscrete& d);
RewardMatrix make_reward_matrix(const madpwrapper::MADPDecPOMDPDiscrete& d,
                                bool use_final_reward);
}

}  // namespace pgi

#endif  // MADPWRAPPERUTILS_H
