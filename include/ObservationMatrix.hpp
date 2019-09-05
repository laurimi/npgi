// ObservationMatrix.hpp
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

#ifndef OBSERVATIONMATRIX_HPP
#define OBSERVATIONMATRIX_HPP
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>
#include "EigenUtils.hpp"
#include "ObservationInterface.h"

namespace pgi {
template <bool isSparse>
class ObservationMatrix : public ObservationInterface {
 public:
  ObservationMatrix(std::size_t num_observations, std::size_t num_states,
                    std::size_t num_actions)
      : ObservationProbabilityMatrix_(
            num_actions, MatrixType(num_observations, num_states)) {}

  void set(std::size_t j_obs, std::size_t state, std::size_t j_act,
           double p) override {
    ObservationProbabilityMatrix_.at(j_act).coeffRef(j_obs, state) = p;
  }
  double get(std::size_t j_obs, std::size_t state,
             std::size_t j_act) const override {
    return ObservationProbabilityMatrix_.at(j_act).coeff(j_obs, state);
  }

  double update(Eigen::VectorXd& state_probabilities, std::size_t j_act,
                std::size_t j_obs) const override {
    return detail::bayes_update(
        state_probabilities,
        ObservationProbabilityMatrix_.at(j_act).row(j_obs).transpose());
  }

  double update(Eigen::SparseVector<double>& state_probabilities, std::size_t j_act,
                std::size_t j_obs) const override {
    return detail::bayes_update(
        state_probabilities,
        ObservationProbabilityMatrix_.at(j_act).row(j_obs).transpose());
  }

  std::size_t sample_observation(std::size_t new_state, std::size_t j_act,
                                 double random01) const override {
    return detail::sample_from_pmf(
        ObservationProbabilityMatrix_.at(j_act).col(new_state), random01);
  }

 private:
  typedef typename std::conditional<isSparse, Eigen::SparseMatrix<double>,
                                    Eigen::MatrixXd>::type MatrixType;
  std::vector<MatrixType> ObservationProbabilityMatrix_;
};

}  // namespace pgi

#endif  // OBSERVATIONMATRIX_HPP
