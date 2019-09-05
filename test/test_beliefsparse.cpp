// test_beliefsparse.cpp
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
#include "gtest/gtest.h"

TEST(BeliefSparseTest, ZeroEntropy) {
  Eigen::SparseVector<double> a(2);
  a.coeffRef(0) = 1.0;
  a.coeffRef(1) = 0.0;

  pgi::belief_t s(a);
  EXPECT_EQ(entropy(s), 0.0);


  Eigen::SparseVector<double> b(100);
  EXPECT_EQ(entropy(pgi::belief_t(b)), 0.0);
}

TEST(BeliefSparseTest, MaxEntropy) {
  Eigen::SparseVector<double> a(2);
  a.coeffRef(0) = 0.5;
  a.coeffRef(1) = 0.5;

  EXPECT_EQ(entropy(pgi::belief_t(a)), 1.0);
}

TEST(BeliefSparseTest, Sampling)
{
  Eigen::SparseVector<double> a(4);
  a.coeffRef(0) = 0.25;
  a.coeffRef(1) = 0.25;
  a.coeffRef(2) = 0.25;
  a.coeffRef(3) = 0.25;

  pgi::belief_t b(a);

  EXPECT_EQ(sample_state(b, 0.0), 0);
  EXPECT_EQ(sample_state(b, 1.0), sample_space_size(b)-1);

  EXPECT_EQ(sample_state(b, 0.2), 0);
  EXPECT_EQ(sample_state(b, 0.4), 1);
  EXPECT_EQ(sample_state(b, 0.6), 2);
  EXPECT_EQ(sample_state(b, 0.8), 3);
}
