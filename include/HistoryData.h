// HistoryData.h
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

#ifndef HISTORYDATA_H
#define HISTORYDATA_H
#include "Belief.hpp"
#include "DecPOMDPDiscrete.h"
#include "History.h"

namespace pgi {
struct HistoryData {
  double probability_ {0.0};
  double last_observation_probability_ {0.0};
  double sum_of_expected_rewards_ {0.0};
  belief_t belief_ {Eigen::VectorXd(0)};
};

bool is_reachable(const HistoryData& d);
HistoryData get_next(const HistoryData& current,
                     const DecPOMDPDiscrete& decpomdp,
                     const ActionObservation& joint, bool add_final_reward);

}  // namespace pgi
#endif  // HISTORYDATA_H
