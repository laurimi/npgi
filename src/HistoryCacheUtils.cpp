// HistoryCacheUtils.cpp
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

#include "HistoryCacheUtils.h"

namespace pgi {
HistoryData get_data_or_insert_missing(
    const History& target, HistoryCache& hc,
    const DecPOMDPDiscrete& decpomdp, bool final_reward_at_end) {
  HistoryCache::tree_vertex_t v;
  History::const_iterator hit;
  std::tie(v, hit) = hc.find_longest_subkey(target);
  if (hit != target.end()) {
    for (; hit != target.end(); ++hit) {
      HistoryData hd = hc.get_data(v);
      bool use_final_reward =
          (hit + 1 == target.end() ? final_reward_at_end : false);
      std::tie(v, std::ignore) =
          hc.add_vertex(v, *hit, get_next(hd, decpomdp, *hit, use_final_reward));
    }
  }
  return hc.get_data(v);
}
}  // namespace pgi
