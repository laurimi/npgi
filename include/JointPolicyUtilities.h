// JointPolicyUtilities.h
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

#ifndef JOINTPOLICY_UTILITIES_H
#define JOINTPOLICY_UTILITIES_H
#include <vector>
#include "Belief.hpp"
#include "DecPOMDPDiscrete.h"
#include "PRNG.h"
#include "PolicyGraph.h"

namespace pgi {
unsigned int find_max_steps(const std::vector<PolicyGraph>& locals);
std::vector<std::vector<vertex_t> > vertices_with_n_steps(
    unsigned int n, const std::vector<PolicyGraph>& locals);

std::vector<vertex_t> find_root(const std::vector<PolicyGraph>& locals);

std::vector<std::size_t> local_actions(const std::vector<vertex_t>& q,
                                       const std::vector<PolicyGraph>& locals);
// set to special policies
std::size_t best_greedy_joint_action(belief_t belief,
                                     const DecPOMDPDiscrete& decpomdp,
                                     bool is_final_step);
void set_open_loop_greedy(std::vector<PolicyGraph>& locals,
                          belief_t initial_belief,
                          const DecPOMDPDiscrete& decpomdp);
void set_blind(std::vector<PolicyGraph>& locals, std::size_t joint_action,
               const JointActionSpace& jas);
void set_random(std::vector<PolicyGraph>& locals, PRNG& rng);
}  // namespace pgi

#endif  // JOINTPOLICY_UTILITIES_H
