// DecPOMDPUtilities.h
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

#ifndef DECPOMDPUTILITIES_H
#define DECPOMDPUTILITIES_H
#include <vector>
#include "Belief.hpp"
#include "DecPOMDPDiscrete.h"
#include "History.h"
#include "JointPolicy.h"
#include "PRNG.h"

namespace pgi {

struct DecPOMDPHistory {
  std::vector<std::size_t> states_;
  History h_;
};

DecPOMDPHistory sample_random_joint_history(
    const DecPOMDPDiscrete& decpomdp, const belief_t& b0,
    PRNG& rng, unsigned int length);

std::pair<DecPOMDPHistory, JointPolicy::joint_vertex_t> sample_joint_history(
    const DecPOMDPDiscrete& decpomdp, const JointPolicy& jp,
    const belief_t& b0, PRNG& rng, unsigned int length);

}  // namespace pgi

#endif  // DECPOMDPUTILITIES_H
