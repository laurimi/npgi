// IndexSpace.hpp
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

#ifndef INDEXSPACE_HPP
#define INDEXSPACE_HPP
#include <functional>
#include <numeric>
#include <vector>

namespace pgi {

template <typename Index>
class IndexSpace {
 public:
  typedef typename std::vector<Index>::size_type vec_size_t;

  virtual ~IndexSpace() = default;
  IndexSpace() : stepsize_(), sz_(0) {}
  IndexSpace(const std::vector<Index>& num_elements)
      : num_elements_(num_elements),
        stepsize_(get_stepsize(num_elements)),
        sz_(std::accumulate(std::begin(num_elements), std::end(num_elements), 1,
                            std::multiplies<vec_size_t>())) {}

  Index num_joint_indices() const { return sz_; }
  Index local_index_size() const { return num_elements_.size(); }
  Index num_local_indices(Index i) const { return num_elements_[i]; }

  Index joint_index(const std::vector<Index>& indices) const {
    Index joint_index = 0;
    for (vec_size_t i = 0; i < stepsize_.size(); ++i)
      joint_index += indices[i] * stepsize_[i];

    return joint_index;
  }

  Index local_index(Index joint_index, std::size_t local_order) const {
    Index local_index = joint_index;
    for (vec_size_t i = 0; i <= local_order; ++i) {
      local_index = joint_index / stepsize_[i];
      joint_index -= stepsize_[i] * local_index;
    }
    return local_index;
  }

 private:
  std::vector<Index> num_elements_;
  std::vector<Index> stepsize_;
  Index sz_;

  static std::vector<Index> get_stepsize(
      const std::vector<Index>& num_elements) {
    const vec_size_t sz = num_elements.size();
    // increment indicates for each agent how many the joint index is
    // incremented to get the next individual action...
    std::vector<Index> step_size(sz);
    if (sz == 0) return step_size;

    // the step_size for the last agent is 1
    step_size[sz - 1] = 1;
    if (sz != 1) {
      vec_size_t i = sz - 2;
      while (1) {
        if (i > 0) {
          step_size[i] = num_elements[i + 1] * step_size[i + 1];
          i--;
        } else if (i == 0) {
          step_size[i] = num_elements[i + 1] * step_size[i + 1];
          break;
        }
      }
    }
    return step_size;
  }
};

}  // namespace pgi
#endif  // INDEXSPACE_HPP
