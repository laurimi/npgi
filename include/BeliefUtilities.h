// BeliefUtilities.h
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

#ifndef BELIEFUTILITIES_HPP
#define BELIEFUTILITIES_HPP
#include <memory>
#include <utility>
#include "Belief.hpp"
#include "HistoryCache.hpp"
#include "DecPOMDPDiscrete.h"
#include "PolicyGraph.h"
#include "JointPolicy.h"

namespace pgi {
std::pair<belief_t, double> joint_belief_with_probability(
    JointPolicy::joint_vertex_t q, const JointPolicy& jp,
    const DecPOMDPDiscrete& decpomdp, HistoryCache& c);


std::pair<belief_t, double> expected_belief_with_probability(
    const std::vector<History>& histories, HistoryCache& c, const DecPOMDPDiscrete& decpomdp, bool final_reward_at_end);
}  // namespace pgi

#endif  // BELIEFUTILITIES_HPP
