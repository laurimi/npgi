// ValueUtilities.h
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

#ifndef VALUEUTILITIES_H
#define VALUEUTILITIES_H
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/weighted_sum.hpp>
#include "DecPOMDPDiscrete.h"
#include "HistoryCache.hpp"
#include "JointPolicy.h"
#include "JointPolicyHistories.h"
#include "PRNG.h"

namespace pgi {
typedef boost::accumulators::accumulator_set<
    double,
    boost::accumulators::features<boost::accumulators::tag::weighted_sum>,
    double>
    stats_t;
typedef std::map<std::size_t, stats_t> value_map_t;



std::size_t best_value_index(const value_map_t& vm);


namespace local_node_value {
// lower bound for the values of local policies of node qlocal in local policy
// idx_local, evaluated by taking the expectation under expected beliefs of
// joint nodes in qj.
value_map_t lower_bound(const DecPOMDPDiscrete& d, HistoryCache& hc,
                        JointPolicy jp, std::size_t idx_local, vertex_t qlocal,
                        const std::vector<JointPolicy::joint_vertex_t>& qj);

value_map_t exact(const DecPOMDPDiscrete& d,
                  HistoryCache& hc, JointPolicy jp,
                  std::size_t idx_local, vertex_t qlocal,
                  const std::vector<JointPolicy::joint_vertex_t>& qj);

value_map_t heuristic_estimate(const DecPOMDPDiscrete& d,
                               HistoryCache& hc,
                               const JointPolicy& jp, std::size_t idx_local,
                               vertex_t qlocal, PRNG& rng,
                               bool use_random_history);

value_map_t policy_state_value(const DecPOMDPDiscrete& d,
                               JointPolicy jp, const PolicyState& ps,
                               std::size_t idx_local, vertex_t qlocal,
                               HistoryCache& hc);

}  // namespace local_node_value

}  // namespace pgi

#endif  // VALUEUTILITIES_H
