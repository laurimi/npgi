// JointPolicy.cpp
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

#include "JointPolicy.h"
#include "PolicyGraph.h"
#include <algorithm>
#include "JointPolicyUtilities.h"
#include "combinations.hpp"

namespace pgi {
JointPolicy::JointPolicy(const std::vector<PolicyGraph>& locals)
    : vm(), stepmap(), locals_(locals) {
  joint_vertex_t current_joint_vertex = 0;

  for (unsigned int steps = 0; steps <= find_max_steps(locals); ++steps) {
    auto v = vertices_with_n_steps(steps, locals);
    for (const auto& qc : make_combinations(v)) {
      vm.insert(vm_type::value_type(current_joint_vertex, get_combination(qc)));
      stepmap.insert(nm_type::value_type(current_joint_vertex, steps));
      ++current_joint_vertex;
    }
  }
}

JointPolicy::joint_vertex_t JointPolicy::to_joint(
    const std::vector<vertex_t>& q) const {
  return vm.right.at(q);
}

const std::vector<vertex_t>& JointPolicy::to_local(joint_vertex_t q) const {
  return vm.left.at(q);
}

bool JointPolicy::is_terminal(joint_vertex_t q) const
{
  return (steps_remaining(q) == 0);
}

unsigned int JointPolicy::min_steps() const { return 1; }
unsigned int JointPolicy::max_steps() const { return find_max_steps(locals_); }

unsigned int JointPolicy::steps_remaining(joint_vertex_t q) const {
  return stepmap.at(q);
}

std::vector<JointPolicy::joint_vertex_t> JointPolicy::joint_vertices() const {
  std::vector<joint_vertex_t> v;
  for (const auto& mit : stepmap) {
    v.push_back(mit.first);
  }
  return v;
}

std::vector<JointPolicy::joint_vertex_t>
JointPolicy::joint_vertices_with_steps_remaining(unsigned int steps) const {
  std::vector<joint_vertex_t> v;
  for (auto& m : stepmap) {
    if (m.second == steps) v.push_back(m.first);
  }
  return v;
}

JointPolicy::joint_vertex_t JointPolicy::root() const {
  std::vector<vertex_t> vr;
  for (const auto& local : locals_) vr.push_back(find_root(local));
  return to_joint(vr);
}

std::vector<JointPolicy::joint_vertex_t>
JointPolicy::joint_vertices_with_agent_at(std::size_t agent,
                                          vertex_t qlocal) const {
  std::vector<joint_vertex_t> q_all = joint_vertices_with_steps_remaining(
      pgi::steps_remaining(qlocal, locals_[agent]));

  std::vector<joint_vertex_t> q_with_agent_at_local;

  std::copy_if(q_all.begin(), q_all.end(),
               std::back_inserter(q_with_agent_at_local),
               [&, agent, qlocal](joint_vertex_t q) {
                 const auto& ql = to_local(q);
                 return (ql[agent] == qlocal);
               });

  return q_with_agent_at_local;
}

}  // namespace pgi
