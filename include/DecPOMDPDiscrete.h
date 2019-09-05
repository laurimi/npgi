// DecPOMDPDiscrete.h
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

#ifndef DECPOMDPDISCRETE_H
#define DECPOMDPDISCRETE_H
#include <memory>
#include "DiscreteAbstractions.hpp"
#include "ObservationInterface.h"
#include "RewardInterface.h"
#include "StateTransitionInterface.h"

namespace pgi {
class DecPOMDPDiscrete {
 public:
  DecPOMDPDiscrete(StateSpace s, JointActionSpace jas,
                   JointObservationSpace jos,
                   std::unique_ptr<StateTransitionInterface> t,
                   std::unique_ptr<ObservationInterface> o,
                   std::unique_ptr<RewardInterface> r)
      : s_(std::move(s)),
        jas_(std::move(jas)),
        jos_(std::move(jos)),
        t_(std::move(t)),
        o_(std::move(o)),
        r_(std::move(r)) {
    assert(jas_.num_local_spaces() == jos_.num_local_spaces());
  }

  const StateTransitionInterface& transition_model() const { return *t_; }
  const ObservationInterface& observation_model() const { return *o_; }
  const RewardInterface& reward_model() const { return *r_; }
  const StateSpace& state_space() const { return s_; }
  const JointActionSpace& joint_action_space() const { return jas_; }
  const JointObservationSpace& joint_observation_space() const { return jos_; }

  std::size_t num_agents() const { return jas_.num_local_spaces(); }

 private:
  StateSpace s_;
  JointActionSpace jas_;
  JointObservationSpace jos_;

  std::unique_ptr<StateTransitionInterface> t_;
  std::unique_ptr<ObservationInterface> o_;
  std::unique_ptr<RewardInterface> r_;
};
}  // namespace pgi

#endif  // DECPOMDPDISCRETE_H
