// PolicyGraphTraversal.h
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

#ifndef POLICYGRAPHTRAVERSAL_H
#define POLICYGRAPHTRAVERSAL_H
#include "GraphTraversal.hpp"
#include "JointPolicy.h"
#include "PolicyGraph.h"

namespace pgi {
class PolicyGraphTraversal {
 public:
  typedef typename boost::graph_traits<PolicyGraph>::vertex_descriptor vertex_t;
  PolicyGraphTraversal(vertex_t vstart, const PolicyGraph& g);
  vertex_t current_vertex() const;
  std::size_t current_action() const;
  bool last_action() const;
  void return_to_previous();
  bool traverse(std::size_t local_observation);
  bool can_traverse() const;

 private:
  const PolicyGraph& g_;
  GraphTraversal<PolicyGraph> tr_;
};

class JointPolicyGraphTraversal {
 public:
  JointPolicyGraphTraversal(JointPolicy::joint_vertex_t vstart,
                            const JointPolicy& g);
  JointPolicy::joint_vertex_t current_vertex() const;
  std::size_t current_action(const JointActionSpace& jas) const;
  bool last_action() const;
  void return_to_previous();
  bool traverse(std::size_t joint_observation,
                const JointObservationSpace& jos);
  bool can_traverse() const;

 private:
  const JointPolicy& g_;
  std::vector<PolicyGraphTraversal> trv_;
};

}  // namespace pgi

#endif  // POLICYGRAPHTRAVERSAL_H
