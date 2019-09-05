// PolicyGraphTraversal.cpp
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

#include "PolicyGraphTraversal.h"

namespace pgi {

PolicyGraphTraversal::PolicyGraphTraversal(vertex_t vstart,
                                           const PolicyGraph& g)
    : g_(g), tr_(vstart, g) {}

typename PolicyGraphTraversal::vertex_t PolicyGraphTraversal::current_vertex()
    const {
  return tr_.current_vertex();
}
std::size_t PolicyGraphTraversal::current_action() const {
  return tr_.current_vertex_properties();
}

bool PolicyGraphTraversal::last_action() const {
  return (steps_remaining(current_vertex(), g_) == 1);
}
void PolicyGraphTraversal::return_to_previous() { tr_.return_to_previous(); }

bool PolicyGraphTraversal::traverse(std::size_t local_observation) {
  auto ct = tr_.can_traverse(local_observation);
  if (ct.second) {
    tr_.traverse(*ct.first);
  }
  return ct.second;
}

bool PolicyGraphTraversal::can_traverse() const { return tr_.can_traverse(); }

JointPolicyGraphTraversal::JointPolicyGraphTraversal(
    JointPolicy::joint_vertex_t vstart, const JointPolicy& g)
    : g_(g), trv_([vstart, &g] {
        const std::vector<vertex_t>& localstart = g.to_local(vstart);
        std::vector<PolicyGraphTraversal> trv;
        for (std::size_t i = 0; i < g.num_agents(); ++i) {
          trv.emplace_back(
              PolicyGraphTraversal(localstart[i], g.local_policy(i)));
        }
        return trv;
      }()) {}

JointPolicy::joint_vertex_t JointPolicyGraphTraversal::current_vertex() const {
  std::vector<vertex_t> q(trv_.size());
  for (std::size_t i = 0; i < trv_.size(); ++i) q[i] = trv_[i].current_vertex();
  return g_.to_joint(q);
}

std::size_t JointPolicyGraphTraversal::current_action(
    const JointActionSpace& jas) const {
  std::vector<std::size_t> a(trv_.size());
  for (std::size_t i = 0; i < trv_.size(); ++i) a[i] = trv_[i].current_action();
  return jas.joint_index(a);
}

bool JointPolicyGraphTraversal::last_action() const {
  return std::any_of(
      trv_.begin(), trv_.end(),
      [](const PolicyGraphTraversal& tv) { return tv.last_action(); });
}

void JointPolicyGraphTraversal::return_to_previous() {
  std::for_each(trv_.begin(), trv_.end(),
                [](PolicyGraphTraversal& tv) { tv.return_to_previous(); });
}

bool JointPolicyGraphTraversal::traverse(std::size_t joint_observation,
                                         const JointObservationSpace& jos) {
  if (!can_traverse()) return false;
  bool success = true;
  std::size_t i = 0;
  for (; i < trv_.size(); ++i) {
    if (!trv_[i].traverse(jos.local_index(joint_observation, i))) {
      success = false;
      break;
    }
  }

  if (!success) {  // undo if failed
    for (std::size_t jb = 0; jb < i; ++jb) trv_[i].return_to_previous();
  }

  return success;
}

bool JointPolicyGraphTraversal::can_traverse() const {
  return std::all_of(
      trv_.begin(), trv_.end(),
      [](const PolicyGraphTraversal& tv) { return tv.can_traverse(); });
}

}  // namespace pgi
