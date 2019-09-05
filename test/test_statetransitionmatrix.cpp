// test_statetransitionmatrix.cpp
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

#include "StateTransitionMatrix.hpp"
#include "gtest/gtest.h"

class StateTransitionMatrixTest : public ::testing::Test {
 protected:
  StateTransitionMatrixTest()
      : flip_flop_dense(num_states, num_actions),
        flip_flop_sparse(num_states, num_actions) {}
  void SetUp() override {
    flip_flop_dense.set(1, 1, 0, 0.0);
    flip_flop_dense.set(0, 1, 0, 1.0);
    flip_flop_dense.set(1, 0, 0, 1.0);
    flip_flop_dense.set(0, 0, 0, 0.0);

    flip_flop_sparse.set(1, 1, 0, 0.0);
    flip_flop_sparse.set(0, 1, 0, 1.0);
    flip_flop_sparse.set(1, 0, 0, 1.0);
    flip_flop_sparse.set(0, 0, 0, 0.0);
  }

  // void TearDown() override {}

  pgi::StateTransitionMatrix<false> flip_flop_dense;
  pgi::StateTransitionMatrix<true> flip_flop_sparse;

  constexpr static std::size_t num_actions = 1;
  constexpr static std::size_t num_states = 2;
};

TEST_F(StateTransitionMatrixTest, DensePriorProbabilityDenseInput) {
  Eigen::VectorXd state_probs(2);
  state_probs << 0.0, 1.0;
  flip_flop_dense.predict(state_probs, 0);
  EXPECT_EQ(state_probs(0), 1.0);
  EXPECT_EQ(state_probs(1), 0.0);
}

TEST_F(StateTransitionMatrixTest, SparsePriorProbabilityDenseInput) {
  Eigen::VectorXd state_probs(2);
  state_probs << 0.0, 1.0;
  flip_flop_sparse.predict(state_probs, 0);
  EXPECT_EQ(state_probs(0), 1.0);
  EXPECT_EQ(state_probs(1), 0.0);
}

TEST_F(StateTransitionMatrixTest, DensePriorProbabilitySparseInput) {
  Eigen::SparseVector<double> state_probs(2);
  state_probs.coeffRef(1) = 1.0;

  flip_flop_dense.predict(state_probs, 0);

  std::vector<double> expected({1.0, 0.0});
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(state_probs.coeff(i), expected.at(i));
  }
}

TEST_F(StateTransitionMatrixTest, SparsePriorProbabilitySparseInput) {
  Eigen::SparseVector<double> state_probs(2);
  state_probs.coeffRef(1) = 1.0;

  flip_flop_sparse.predict(state_probs, 0);

  std::vector<double> expected({1.0, 0.0});
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(state_probs.coeff(i), expected.at(i));
  }
}
