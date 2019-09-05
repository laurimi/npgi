// DecPOMDPUtilities.cpp
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

#include "DecPOMDPUtilities.h"
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int.hpp>
#include "PolicyGraphTraversal.h"

namespace pgi {
DecPOMDPHistory sample_random_joint_history(const DecPOMDPDiscrete& decpomdp,
                                            const belief_t& b0, PRNG& rng,
                                            unsigned int length) {
  std::vector<std::size_t> states;
  boost::random::uniform_01<double> dist01;
  states.emplace_back(sample_state(b0, rng(dist01)));

  if (length == 0) return DecPOMDPHistory{{states}, History()};

  History h;
  h.reserve(length);
  boost::random::uniform_int_distribution<std::size_t> act_dist(
      0, decpomdp.joint_action_space().num_joint_indices() - 1);
  for (unsigned int i = 0; i < length; ++i) {
    const std::size_t j_act = rng(act_dist);
    states.emplace_back(decpomdp.transition_model().sample_next_state(
        states.back(), j_act, rng(dist01)));
    const std::size_t j_obs = decpomdp.observation_model().sample_observation(
        states.back(), j_act, rng(dist01));
    h.emplace_back(ActionObservation{j_act, j_obs});
  }
  return DecPOMDPHistory{{states}, h};
}

std::pair<DecPOMDPHistory, JointPolicy::joint_vertex_t> sample_joint_history(
    const DecPOMDPDiscrete& decpomdp, const JointPolicy& jp, const belief_t& b0,
    PRNG& rng, unsigned int length) {
  JointPolicyGraphTraversal tv(jp.root(), jp);
  std::vector<std::size_t> states;
  boost::random::uniform_01<double> dist01;
  states.emplace_back(sample_state(b0, rng(dist01)));

  if (length == 0)
    return std::make_pair(DecPOMDPHistory{{states}, History()},
                          tv.current_vertex());

  History h;
  h.reserve(length);

  const JointActionSpace jas = decpomdp.joint_action_space();
  const JointObservationSpace jos = decpomdp.joint_observation_space();

  for (unsigned int i = 0; i < length; ++i) {
    const std::size_t j_act = tv.current_action(jas);
    states.emplace_back(decpomdp.transition_model().sample_next_state(
        states.back(), j_act, rng(dist01)));
    const std::size_t j_obs = decpomdp.observation_model().sample_observation(
        states.back(), j_act, rng(dist01));
    h.emplace_back(ActionObservation{j_act, j_obs});
    tv.traverse(j_obs, decpomdp.joint_observation_space());
  }
  return std::make_pair(DecPOMDPHistory{{states}, h}, tv.current_vertex());
}

}  // namespace pgi
