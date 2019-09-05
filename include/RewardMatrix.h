// RewardMatrix.h
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

#ifndef REWARDMATRIX_H
#define REWARDMATRIX_H
#include <eigen3/Eigen/Dense>
#include <vector>
#include "RewardInterface.h"
#include "EigenUtils.hpp"

namespace pgi {
class RewardMatrix : public RewardInterface {
 public:
  RewardMatrix(std::size_t num_states, std::size_t num_actions,
               bool use_final_reward)
      : RewardVector_(num_actions, Eigen::VectorXd(num_states)),
        use_final_reward_(use_final_reward) {}
  void set(std::size_t state, std::size_t j_act, double reward) override {
    RewardVector_.at(j_act)(state) = reward;
  }
  double get(std::size_t state, std::size_t j_act) const override {
    return RewardVector_.at(j_act)(state);
  }

  double final_reward(
      const Eigen::VectorXd& state_probabilities) const override {
    if (!use_final_reward_)
      return 0.0;
    else
      return -pgi::detail::entropy(state_probabilities);
  };
  double final_reward(
      const Eigen::SparseVector<double>& state_probabilities) const override {
    if (!use_final_reward_)
      return 0.0;
    else
      return -pgi::detail::entropy(state_probabilities);
  };
  double reward(const Eigen::VectorXd& state_probabilities,
                std::size_t j_act) const override {
    return RewardVector_.at(j_act).dot(state_probabilities);
  }

  double reward(const Eigen::SparseVector<double>& state_probabilities,
                std::size_t j_act) const override {
    return RewardVector_.at(j_act).sparseView().dot(state_probabilities);
  }

 private:
  std::vector<Eigen::VectorXd> RewardVector_;
  bool use_final_reward_;
};
}

#endif  // REWARDMATRIX_H
