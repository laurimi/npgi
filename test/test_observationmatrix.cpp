// test_observationmatrix.cpp
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

#include "Belief.hpp"
#include "ObservationMatrix.hpp"
#include "gtest/gtest.h"

class ObservationMatrixTest : public ::testing::Test {
 protected:
  ObservationMatrixTest()
      : tiger_obs_dense(num_observations, num_states, num_actions),
        tiger_obs_sparse(num_observations, num_states, num_actions) {}
  void SetUp() override {
    tiger_obs_dense.set(0, 0, 0, 0.85);
    tiger_obs_dense.set(1, 0, 0, 0.15);
    tiger_obs_dense.set(0, 1, 0, 0.15);
    tiger_obs_dense.set(1, 1, 0, 0.85);

    tiger_obs_sparse.set(0, 0, 0, 0.85);
    tiger_obs_sparse.set(1, 0, 0, 0.15);
    tiger_obs_sparse.set(0, 1, 0, 0.15);
    tiger_obs_sparse.set(1, 1, 0, 0.85);
  }

  // void TearDown() override {}

  pgi::ObservationMatrix<false> tiger_obs_dense;
  pgi::ObservationMatrix<true> tiger_obs_sparse;

  constexpr static std::size_t num_actions = 1;  // listen only
  constexpr static std::size_t num_observations = 2;
  constexpr static std::size_t num_states = 2;
};

TEST_F(ObservationMatrixTest, DenseDense) {
  Eigen::VectorXd state_probs(2);
  state_probs << 0.0, 1.0;
  std::size_t obs = 1;
  double p_obs = tiger_obs_dense.update(state_probs, 0, obs);
  EXPECT_EQ(p_obs, 0.85);

  EXPECT_EQ(state_probs(0), 0.0);
  EXPECT_EQ(state_probs(1), 1.0);
}

TEST_F(ObservationMatrixTest, DenseSparse) {
  Eigen::VectorXd state_probs(2);
  state_probs << 0.0, 1.0;
  std::size_t obs = 1;
  double p_obs = tiger_obs_sparse.update(state_probs, 0, obs);
  EXPECT_EQ(p_obs, 0.85);
  EXPECT_EQ(state_probs(0), 0.0);
  EXPECT_EQ(state_probs(1), 1.0);
}

TEST_F(ObservationMatrixTest, SparseDense) {
  Eigen::SparseVector<double> state_probs(2);
  state_probs.coeffRef(1) = 1.0;

  std::size_t obs = 1;
  double p_obs = tiger_obs_dense.update(state_probs, 0, obs);
  EXPECT_EQ(p_obs, 0.85);

  std::vector<double> expected({0.0, 1.0});
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(state_probs.coeff(i), expected.at(i));
  }
}

TEST_F(ObservationMatrixTest, SparseSparse) {
  Eigen::SparseVector<double> state_probs(2);
  state_probs.coeffRef(1) = 1.0;

  std::size_t obs = 1;
  double p_obs = tiger_obs_sparse.update(state_probs, 0, obs);
  EXPECT_EQ(p_obs, 0.85);

  std::vector<double> expected{0.0, 1.0};
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(state_probs.coeff(i), expected.at(i));
  }
}
