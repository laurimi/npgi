// JointPolicy.h
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

#ifndef JOINTPOLICY_H
#define JOINTPOLICY_H
#include <vector>
#include <map>
#include <boost/bimap.hpp>
#include "PolicyGraph.h"
namespace pgi {
class JointPolicy {
 public:
  typedef vertex_t joint_vertex_t;
  JointPolicy(const std::vector<PolicyGraph>& locals);

  joint_vertex_t to_joint(const std::vector<vertex_t>& q) const;
  const std::vector<vertex_t>& to_local(joint_vertex_t q) const;
  const std::vector<PolicyGraph>& local_policies() const { return locals_; }

  bool is_terminal(joint_vertex_t q) const;

  std::size_t num_agents() const { return locals_.size(); }
  const PolicyGraph& local_policy(std::size_t i) const { return locals_[i]; }

  PolicyGraph& local_policy(std::size_t i) { return locals_[i]; }


  unsigned int min_steps() const;
  unsigned int max_steps() const;

  unsigned int steps_remaining(joint_vertex_t q) const;
  std::vector<joint_vertex_t> joint_vertices() const;
  std::vector<joint_vertex_t> joint_vertices_with_steps_remaining(
      unsigned int steps) const;
  std::vector<joint_vertex_t> joint_vertices_with_agent_at(std::size_t agent, vertex_t qlocal) const;
  joint_vertex_t root() const;

 private:
  typedef boost::bimap<joint_vertex_t, std::vector<vertex_t> > vm_type;
  typedef std::map<joint_vertex_t, unsigned int> nm_type;
  vm_type vm;
  nm_type stepmap;
  std::vector<PolicyGraph> locals_;
};
}  // namespace pgi

#endif  // JOINTPOLICY_H
