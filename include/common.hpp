// common.hpp
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

#ifndef COMMON_HPP
#define COMMON_HPP
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace pgi {
template <typename T>
bool is_almost_zero(T x) {
  return (std::abs(x) < std::numeric_limits<T>::epsilon());
}
}  // namespace pgi

template <typename T, std::size_t N>
std::array<std::size_t, N> sort_indices(const std::array<T, N>& v) {
  // initialize original index locations
  std::array<std::size_t, N> idx;
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return idx;
}


#endif  // COMMON_HPP
