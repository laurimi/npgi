// LocalPolicyHistory.cpp
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

#include "LocalPolicyHistory.h"
#include <fstream>
#include <iostream>
#include <sstream>
namespace pgi {
LocalPolicyHistory::LocalPolicyHistory(const std::string& filename)
    : observation_history_to_action_(),
      max_steps_(1) {
  std::ifstream fs(filename);
  std::string line;
  while (getline(fs, line)) {
    std::istringstream iss(line);
    std::vector<unsigned int> v;
    unsigned int number;
    while (iss >> number) {
      v.push_back(number);
    }

    unsigned int action_index = v.back();
    v.pop_back();

    observation_history_to_action_[v] = action_index;

    if (v.size() > max_steps_) max_steps_ = v.size();
  }
}

std::pair<std::size_t, bool> LocalPolicyHistory::next_action_index(
    const History& h) const {
  std::vector<unsigned int> obs_history;
  for (const auto& e : h) {
    obs_history.emplace_back(e.observation_index_);
  }

  auto it = observation_history_to_action_.find(obs_history);
  if (it == observation_history_to_action_.end())
    return std::make_pair(0, false);
  else
    return std::make_pair(it->second, true);
}
}
