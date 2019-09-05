// RewardModel.hpp
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

#ifndef REWARDMODEL_HPP
#define REWARDMODEL_HPP
#include "NPGICRTP.hpp"
#include <vector>

namespace pgi {
template <typename State>
class RewardModel : crtp<State> {
 public:
  double get(const std::vector<State>& states,
             const std::vector<double>& weights, std::size_t j_act) const {
    return this->underlying().get(states, weights, j_act);
  }
  double final_reward(const std::vector<State>& states,
                      const std::vector<double>& weights) const {
    return this->underlying().final_reward(states, weights);
  }
};
}  // namespace pgi

#endif  // REWARDMODEL_HPP
