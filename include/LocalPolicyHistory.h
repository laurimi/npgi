// LocalPolicyHistory.h
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

#ifndef LOCALPOLICYHISTORY_H
#define LOCALPOLICYHISTORY_H
#include <map>
#include <tuple>
#include "History.h"

namespace pgi {
class LocalPolicyHistory {
 public:
  LocalPolicyHistory(const std::string& filename);
  unsigned int min_steps() const { return 1; }
  unsigned int max_steps() const { return max_steps_; }
  std::pair<std::size_t, bool> next_action_index(const History& local) const;

 private:
  std::map<std::vector<unsigned int>, std::size_t>
      observation_history_to_action_;
  unsigned int max_steps_;
};
}  // namespace pgi

#endif  // LOCALPOLICYHISTORY_H
