// StateTransitionMatrix.hpp
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

#ifndef STATETRANSITIONMATRIX_HPP
#define STATETRANSITIONMATRIX_HPP
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>
#include "EigenUtils.hpp"
#include "StateTransitionInterface.h"

namespace pgi {
template <bool isSparse>
class StateTransitionMatrix : public StateTransitionInterface {
 public:
  StateTransitionMatrix(std::size_t num_states, std::size_t num_actions)
      : StateTransitionProbabilityMatrix_(num_actions,
                                          MatrixType(num_states, num_states)) {}

  void set(std::size_t next_state, std::size_t state, std::size_t j_act,
           double p) override {
    StateTransitionProbabilityMatrix_.at(j_act).coeffRef(next_state, state) = p;
  }
  double get(std::size_t next_state, std::size_t state,
             std::size_t j_act) const override {
    return StateTransitionProbabilityMatrix_.at(j_act).coeff(next_state, state);
  }

  void predict(Eigen::SparseVector<double>& state_probabilities,
               std::size_t j_act) const override {
    detail::inplace_matmul(state_probabilities,
                           StateTransitionProbabilityMatrix_.at(j_act));
  }

  void predict(Eigen::VectorXd& state_probabilities,
               std::size_t j_act) const override {
    detail::inplace_matmul(state_probabilities,
                           StateTransitionProbabilityMatrix_.at(j_act));
  }

  Eigen::VectorXd predicted(const Eigen::VectorXd& state_probabilities,
                            std::size_t j_act) const override {
    Eigen::VectorXd predicted = state_probabilities;
    predict(predicted, j_act);
    return predicted;
  }
  Eigen::SparseVector<double> predicted(
      const Eigen::SparseVector<double>& state_probabilities,
      std::size_t j_act) const override {
    Eigen::SparseVector<double> predicted = state_probabilities;
    predict(predicted, j_act);
    return predicted;
  }

  std::size_t sample_next_state(std::size_t state, std::size_t j_act,
                                double random01) const override {
    return detail::sample_from_pmf(
        StateTransitionProbabilityMatrix_.at(j_act).col(state), random01);
  }

 private:
  typedef typename std::conditional<isSparse, Eigen::SparseMatrix<double>,
                                    Eigen::MatrixXd>::type MatrixType;
  std::vector<MatrixType> StateTransitionProbabilityMatrix_;
};

}  // namespace pgi

#endif  // STATETRANSITIONMATRIX_HPP
