// JointPolicyHistories.h
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

#ifndef JOINTPOLICYHISTORIES_H
#define JOINTPOLICYHISTORIES_H
#include <vector>
#include "History.h"
#include "JointPolicy.h"
#include "PolicyGraph.h"
#include "Belief.hpp"
#include "PRNG.h"

namespace pgi {
class PathReporter {
 public:
  typedef std::vector<History> HistoryVector;
  PathReporter(HistoryVector* m, const PolicyGraph& g) : m_(m), g_(g) {}
  void operator()(const path_t& p) const {
    (*m_).emplace_back(get_history(p, g_));
  }

 private:
  HistoryVector* m_;
  const PolicyGraph& g_;
};

struct PolicyState {
  History history_;
  JointPolicy::joint_vertex_t joint_vertex_;
};

std::vector<std::vector<History>> local_histories_at(const std::vector<vertex_t>& qv, const JointPolicy& jp);
std::pair<History, bool> sample_joint_history_from_policy(
    JointPolicy::joint_vertex_t q, const JointPolicy& jp,
    const JointActionSpace& jas, const JointObservationSpace& jos, PRNG& rng);
    
std::pair<PolicyState, bool> sample_history_and_joint_vertex_from_policy(
    const JointActionSpace& jas, const JointObservationSpace& jos,
    const JointPolicy& jp, unsigned int steps_remaining, PRNG& rng);

std::pair<PolicyState, bool> sample_random_history_and_joint_vertex(
    const DecPOMDPDiscrete& d, const belief_t& b0,
    const JointPolicy& jp, std::size_t idx_local, vertex_t qlocal, PRNG& rng);

}  // namespace pgi

#endif  // JOINTPOLICYHISTORIES_H
