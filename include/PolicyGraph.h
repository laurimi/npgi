// PolicyGraph.h
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

#ifndef POLICYGRAPH_H
#define POLICYGRAPH_H
#include <boost/function.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <iostream>
#include <vector>
#include "History.h"
#include "PRNG.h"

namespace pgi {
struct PGProperties {
  std::size_t num_actions_;
  std::size_t num_observations_;
  unsigned int num_steps_;
};

std::size_t num_edge_configs(unsigned int step, std::size_t target_width,
                             const PGProperties& properties);
std::size_t num_local_policies(unsigned int step, std::size_t target_width,
                               const PGProperties& properties);
std::size_t layer_width(unsigned int step, std::size_t target_width,
                        const PGProperties& properties);

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                              std::size_t, std::size_t, PGProperties>
    PolicyGraph;
typedef typename boost::graph_traits<PolicyGraph>::vertex_descriptor vertex_t;
typedef typename boost::graph_traits<PolicyGraph>::edge_descriptor edge_t;
typedef std::vector<edge_t> path_t;
typedef boost::function<bool(vertex_t x)> vertex_predicate_t;

History get_history(const path_t& p, const PolicyGraph& g);

PolicyGraph fixed_width(unsigned int num_steps, std::size_t target_width,
                        std::size_t num_actions, std::size_t num_observations);

unsigned int steps_remaining(vertex_t v, const PolicyGraph& g);

vertex_t find_root(const PolicyGraph& g);
std::pair<typename boost::filter_iterator<
              vertex_predicate_t,
              typename boost::graph_traits<PolicyGraph>::vertex_iterator>,
          typename boost::filter_iterator<
              vertex_predicate_t,
              typename boost::graph_traits<PolicyGraph>::vertex_iterator> >
vertices_with_steps_remaining(unsigned int s, const PolicyGraph& g);
std::pair<typename boost::filter_iterator<
              vertex_predicate_t,
              typename boost::graph_traits<PolicyGraph>::vertex_iterator>,
          typename boost::filter_iterator<
              vertex_predicate_t,
              typename boost::graph_traits<PolicyGraph>::vertex_iterator> >
possible_successors(vertex_t v, const PolicyGraph& g);
std::size_t num_possible_successors(vertex_t v, const PolicyGraph& g);
std::size_t num_out_edge_configurations(vertex_t v, const PolicyGraph& g);
std::size_t num_local_policies(vertex_t v, const PolicyGraph& g);
bool same_policy(vertex_t v, vertex_t u, const PolicyGraph& g);
bool any_same_policy(vertex_t v, const std::vector<vertex_t>& v_compare,
                     const PolicyGraph& g);
bool redirect_in_edges_of_same_policy(vertex_t from,
                                      const std::vector<vertex_t>& to,
                                      PolicyGraph& g, PRNG& rng);
void set_out_edge_configuration(vertex_t v,
                                std::size_t out_edge_configuration_idx,
                                PolicyGraph& g);
void set_local_policy(vertex_t v, std::size_t local_policy_idx, PolicyGraph& g);
std::size_t get_local_policy(vertex_t v, std::size_t action_index,
                             std::size_t out_edge_configuration_idx,
                             const PolicyGraph& g);

std::ostream& print(std::ostream& out, const PolicyGraph& g,
                    const std::vector<std::string>& actionnames,
                    const std::vector<std::string>& observationnames);

std::vector<vertex_t> vertices_with_n_steps(unsigned int n,
                                            const PolicyGraph& local);

void randomize(PolicyGraph& g, PRNG& rng);
void randomize_nodes_with_steps_remaining(unsigned int steps, PolicyGraph& g,
                                          PRNG& rng);
void randomize_local_policy(PolicyGraph& g, vertex_t v, PRNG& rng,
                            const std::vector<vertex_t>& avoid_same_policy);

}  // namespace pgi

#endif  // POLICYGRAPH_H
