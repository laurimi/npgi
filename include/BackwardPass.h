// BackwardPass.h
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

#ifndef BACKWARDPASS_H
#define BACKWARDPASS_H
#include <memory>
#include "DecPOMDPDiscrete.h"
#include "HistoryCache.hpp"
#include "JointPolicy.h"
#include "JointPolicyHistories.h"
#include "PRNG.h"
#include "ValueUtilities.h"

namespace pgi {
namespace backpass {
struct BackPassProperties {
  BackPassProperties() = default;
  BackPassProperties(bool use_lower_bound) : use_lower_bound(use_lower_bound) {}
  double prob_heuristic_improvement{0.5};
  double prob_random_history_in_heuristic_improvement{0.9};
  bool use_lower_bound{false};
};

struct ImprovementResult {
  JointPolicy improved_policy;
  double improved_policy_value;
};

ImprovementResult improve(JointPolicy jp, const DecPOMDPDiscrete& decpomdp,
                          HistoryCache& cache, PRNG& rng,
                          const BackPassProperties& props);

value_map_t estimate_local_policy_values(
    const JointPolicy& jp, const std::size_t agent_idx,
    const vertex_t agent_vertex, const DecPOMDPDiscrete& decpomdp,
    HistoryCache& cache, PRNG& rng, const BackPassProperties& props);

}  // namespace backpass
}  // namespace pgi

#endif  // BACKWARDPASS_H
