// ParticleUtilities.cpp
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

#include "ParticleUtilities.h"
#include <boost/random/uniform_real_distribution.hpp>
#include <eigen3/Eigen/Dense>
#include "common.hpp"

namespace pgi {
double normalize(std::vector<double>& weights) {
  if (weights.empty()) {
    return 0.0;
  } else {
    Eigen::Map<Eigen::VectorXd> w(weights.data(), weights.size());
    const double wsum = w.lpNorm<1>();
    if (!is_almost_zero(wsum)) w /= wsum;
    return wsum;
  }
}

// Implements systematic resampling
void resample(const std::vector<double>& weights,
              std::vector<std::size_t>& resample_indices, PRNG& rng) {
  resample_indices.resize(weights.size());
  double cdf = 0.0;
  std::size_t curr_index = 0;

  const double incr = 1.0 / static_cast<double>(weights.size());
  boost::random::uniform_real_distribution<double> U(0, incr);
  double u = rng(U);
  for (auto& resample_index : resample_indices) {
    while (u > cdf) {
      cdf += weights[curr_index];
      ++curr_index;
    }
    resample_index = curr_index - 1;
    u += incr;
  }
}

double effective_size(const std::vector<double>& weights) {
  Eigen::Map<const Eigen::VectorXd> w(weights.data(), weights.size());
  return 1.0 / w.squaredNorm();
}

}  // namespace pgi
