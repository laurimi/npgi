// History.h
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

#ifndef HISTORY_H
#define HISTORY_H
#include <boost/functional/hash.hpp>
#include <vector>
#include "DecPOMDPDiscrete.h"
#include "PRNG.h"

namespace pgi {
struct ActionObservation {
  std::size_t action_index_;
  std::size_t observation_index_;
  static constexpr std::size_t _MAX_OBSERVATIONS_FOR_HASH =
      10000;  // needed just for the custom hash
};

inline bool operator==(const ActionObservation& a, const ActionObservation& b) {
  return ((a.action_index_ == b.action_index_) &&
          (a.observation_index_ == b.observation_index_));
}

inline std::size_t hash_value(const ActionObservation& ao) {
  boost::hash<std::size_t> hasher;
  return hasher(ActionObservation::_MAX_OBSERVATIONS_FOR_HASH *
                    ao.action_index_ +
                ao.observation_index_);
}

typedef std::vector<ActionObservation> History;

std::size_t joint_action_at_step(const std::vector<History>& locals,
                                 const JointActionSpace& jas, std::size_t step);

std::size_t joint_observation_at_step(const std::vector<History>& locals,
                                      const JointObservationSpace& jos,
                                      std::size_t step);

History local_to_joint_history(const std::vector<History>& locals,
                               const JointActionSpace& jas,
                               const JointObservationSpace& jos);

ActionObservation joint_to_local(const ActionObservation& joint,
                                 const JointObservationSpace& jos,
                                 std::size_t agent_idx);
std::vector<History> joint_to_local_history(
    const History& joint, const DecPOMDPDiscrete& decpomdp);

}  // namespace pgi

#endif
