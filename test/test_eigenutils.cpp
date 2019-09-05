// test_eigenutils.cpp
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

#include "EigenUtils.hpp"
#include "gtest/gtest.h"

TEST(EigenUtilsTest, ZeroEntropy) {
  Eigen::SparseVector<double> b(2);
  b.coeffRef(0) = 0.0;
  b.coeffRef(1) = 1.0;

  EXPECT_EQ(0.0, pgi::detail::entropy(b));

  Eigen::VectorXd c(2);
  c << 0.0, 1.0;
  EXPECT_EQ(0.0, pgi::detail::entropy(c));
}

TEST(EigenUtilsTest, MaxEntropy) {
  Eigen::SparseVector<double> b(2);
  b.coeffRef(0) = 0.5;
  b.coeffRef(1) = 0.5;

  EXPECT_EQ(1.0, pgi::detail::entropy(b));

  Eigen::VectorXd c(2);
  c << 0.5, 0.5;
  EXPECT_EQ(1.0, pgi::detail::entropy(c));
}

TEST(EigenUtilsTest, EmptyEntropy) {
  Eigen::SparseVector<double> b;
  EXPECT_EQ(0.0, pgi::detail::entropy(b));

  Eigen::VectorXd c;
  EXPECT_EQ(0.0, pgi::detail::entropy(c));
}

TEST(EigenUtilsTest, PMFSamplingSparse) {
  Eigen::SparseVector<double> b(4);
  b.coeffRef(0) = 0.25;
  b.coeffRef(1) = 0.25;
  b.coeffRef(2) = 0.25;
  b.coeffRef(3) = 0.25;

  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.0), 0);
  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 1.0), b.size() - 1);

  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.2), 0);
  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.4), 1);
  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.6), 2);
  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.8), 3);
}

TEST(EigenUtilsTest, PMFSamplingDense) {
  Eigen::VectorXd b(4);
  b << 0.25, 0.25, 0.25, 0.25;

  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.0), 0);
  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 1.0), b.size() - 1);

  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.2), 0);
  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.4), 1);
  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.6), 2);
  EXPECT_EQ(pgi::detail::sample_from_pmf(b, 0.8), 3);
}

TEST(EigenUtilsTest, BayesDenseDense) {
  Eigen::VectorXd pmf(4);
  pmf << 0.25, 0.25, 0.25, 0.25;
  Eigen::VectorXd p_obs(4);
  p_obs << 0.5, 0.5, 0.0, 0.0;

  const double prior = pgi::detail::bayes_update(pmf, p_obs);

  EXPECT_EQ(prior, 0.25);
  std::vector<double> expected{0.5, 0.5, 0.0, 0.0};
  for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(pmf.coeff(i), expected[i]);
}

TEST(EigenUtilsTest, BayesDenseSparse) {
  Eigen::VectorXd pmf(4);
  pmf << 0.25, 0.25, 0.25, 0.25;
  Eigen::SparseVector<double> p_obs(4);
  p_obs.coeffRef(0) = 0.5;
  p_obs.coeffRef(1) = 0.5;
  // p_obs.coeffRef(2) = 0.0;
  // p_obs.coeffRef(3) = 0.0;

  const double prior = pgi::detail::bayes_update(pmf, p_obs);

  EXPECT_EQ(prior, 0.25);
  std::vector<double> expected{0.5, 0.5, 0.0, 0.0};
  for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(pmf.coeff(i), expected[i]);
}

TEST(EigenUtilsTest, BayesSparseSparse) {
  Eigen::SparseVector<double> pmf(4);
  pmf.coeffRef(0) = 0.25;
  pmf.coeffRef(1) = 0.25;
  pmf.coeffRef(2) = 0.25;
  pmf.coeffRef(3) = 0.25;
  Eigen::SparseVector<double> p_obs(4);
  p_obs.coeffRef(0) = 0.5;
  p_obs.coeffRef(1) = 0.5;
  // p_obs.coeffRef(2) = 0.0;
  // p_obs.coeffRef(3) = 0.0;

  const double prior = pgi::detail::bayes_update(pmf, p_obs);

  EXPECT_EQ(prior, 0.25);

  std::vector<double> expected{0.5, 0.5, 0.0, 0.0};
  for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(pmf.coeff(i), expected[i]);
}

TEST(EigenUtilsTest, BayesSparseDense) {
  Eigen::SparseVector<double> pmf(4);
  pmf.coeffRef(0) = 0.25;
  pmf.coeffRef(1) = 0.25;
  pmf.coeffRef(2) = 0.25;
  pmf.coeffRef(3) = 0.25;
  Eigen::VectorXd p_obs(4);
  p_obs << 0.5, 0.5, 0.0, 0.0;

  const double prior = pgi::detail::bayes_update(pmf, p_obs);

  EXPECT_EQ(prior, 0.25);
  std::vector<double> expected{0.5, 0.5, 0.0, 0.0};
  for (std::size_t i = 0; i < 4; ++i) EXPECT_EQ(pmf.coeff(i), expected[i]);
}

TEST(EigenUtilsTest, MatMulDenseDense) {
  Eigen::VectorXd x(3);
  x << 1, 2, 3;

  Eigen::MatrixXd A(3, 3);
  A << 1, 1, 1, 2, 2, 2, 3, 3, 3;

  pgi::detail::inplace_matmul(x, A);

  std::vector<double> expected{6, 12, 18};
  for (std::size_t i = 0; i < 3; ++i) EXPECT_EQ(x.coeff(i), expected[i]);
}

TEST(EigenUtilsTest, MatMulDenseSparse) {
  Eigen::VectorXd x(3);
  x << 1, 2, 3;

  Eigen::SparseMatrix<double> A(3, 3);
  A.coeffRef(0, 0) = 1;
  A.coeffRef(1, 0) = 2;
  A.coeffRef(2, 0) = 3;
  A.coeffRef(0, 1) = 1;
  A.coeffRef(1, 1) = 2;
  A.coeffRef(2, 1) = 3;
  A.coeffRef(0, 2) = 1;
  A.coeffRef(1, 2) = 2;
  A.coeffRef(2, 2) = 3;

  pgi::detail::inplace_matmul(x, A);

  std::vector<double> expected{6, 12, 18};
  for (std::size_t i = 0; i < 3; ++i) EXPECT_EQ(x.coeff(i), expected[i]);
}

TEST(EigenUtilsTest, MatMulSparseSparse) {
  Eigen::SparseVector<double> x(3);
  x.coeffRef(0) = 1;
  x.coeffRef(1) = 2;
  x.coeffRef(2) = 3;

  Eigen::SparseMatrix<double> A(3, 3);
  A.coeffRef(0, 0) = 1;
  A.coeffRef(1, 0) = 2;
  A.coeffRef(2, 0) = 3;
  A.coeffRef(0, 1) = 1;
  A.coeffRef(1, 1) = 2;
  A.coeffRef(2, 1) = 3;
  A.coeffRef(0, 2) = 1;
  A.coeffRef(1, 2) = 2;
  A.coeffRef(2, 2) = 3;

  pgi::detail::inplace_matmul(x, A);

  std::vector<double> expected{6, 12, 18};
  for (std::size_t i = 0; i < 3; ++i) EXPECT_EQ(x.coeff(i), expected[i]);
}

TEST(EigenUtilsTest, MatMulSparseDense) {
  Eigen::SparseVector<double> x(3);
  x.coeffRef(0) = 1;
  x.coeffRef(1) = 2;
  x.coeffRef(2) = 3;

  Eigen::MatrixXd A(3, 3);
  A << 1, 1, 1, 2, 2, 2, 3, 3, 3;

  pgi::detail::inplace_matmul(x, A);

  std::vector<double> expected{6, 12, 18};
  for (std::size_t i = 0; i < 3; ++i) EXPECT_EQ(x.coeff(i), expected[i]);
}

TEST(EigenUtilsTest, WeightedCovarianceTest) {
  // DIMS = 2
  Eigen::Matrix<double, 2, Eigen::Dynamic> data(2, 3);
  data << 1.0, 1.0, 2.0, 3.0, 4.0, 5.0;
  Eigen::VectorXd weights(3);
  weights << 0.4, 0.2, 0.4;
  Eigen::MatrixXd result;

  pgi::detail::weighted_cov(data, weights, result);

  EXPECT_EQ(2, result.rows());
  EXPECT_EQ(2, result.cols());

  std::vector<double> expected{0.375, 0.625, 0.625, 1.25};
  for (int i = 0; i < result.size(); ++i)
    EXPECT_NEAR(result.coeff(i), expected[i], 1e-12);
}

TEST(EigenUtilsTest, WeightedCovarianceSingleSample) {
  // DIMS = 2
  Eigen::Matrix<double, 2, Eigen::Dynamic> data(2, 1);
  data << 1.0, 1.0;
  Eigen::VectorXd weights(1);
  weights << 1.0;
  Eigen::MatrixXd result;

  pgi::detail::weighted_cov(data, weights, result);

  EXPECT_EQ(2, result.rows());
  EXPECT_EQ(2, result.cols());

  std::vector<double> expected{0.0, 0.0, 0.0, 0.0};
  for (int i = 0; i < result.size(); ++i)
    EXPECT_NEAR(result.coeff(i), expected[i], 1e-12);
}
