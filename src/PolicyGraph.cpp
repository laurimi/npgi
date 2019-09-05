// PolicyGraph.cpp
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

#include "PolicyGraph.h"
#include <boost/random/uniform_int.hpp>
#include "GraphUtilities.hpp"
#include "IndexSpace.hpp"

namespace pgi {

History get_history(const path_t& p, const PolicyGraph& g) {
  History h;
  h.reserve(p.size());
  for (const auto& e : p) {
    h.emplace_back(ActionObservation{g[boost::source(e, g)], g[e]});
  }
  return h;
}

std::size_t num_edge_configs(unsigned int step, std::size_t target_width,
                             const PGProperties& properties) {
  if (step == 1)
    return 1;
  else
    return std::pow(layer_width(step - 1, target_width, properties),
                    properties.num_observations_);
}

std::size_t num_local_policies(unsigned int step, std::size_t target_width,
                               const PGProperties& properties) {
  return properties.num_actions_ *
         num_edge_configs(step, target_width, properties);
}

std::size_t layer_width(unsigned int step, std::size_t target_width,
                        const PGProperties& properties) {
  if (step == properties.num_steps_)
    return 1;
  else
    return std::min(target_width,
                    num_local_policies(step, target_width, properties));
}

PolicyGraph fixed_width(unsigned int num_steps, std::size_t target_width,
                        std::size_t num_actions, std::size_t num_observations) {
  std::size_t vertex_counter(0);
  typedef std::pair<std::size_t, std::size_t> edge_pair_t;
  std::vector<edge_pair_t> edges;
  std::vector<std::size_t> edge_properties;

  PGProperties properties{num_actions, num_observations, num_steps};

  std::vector<std::size_t> src, trg;
  for (unsigned int s = 1; s <= num_steps; ++s) {
    // old sources become new targets
    std::swap(trg, src);
    // add new source vertices to graph
    src.clear();
    const std::size_t n_src = layer_width(s, target_width, properties);
    std::generate_n(std::back_inserter(src), n_src,
                    [&vertex_counter]() { return vertex_counter++; });

    // add an end node for all last action nodes
    if (s == 1) {
      std::for_each(src.begin(), src.end(),
                    [num_observations, &vertex_counter, &edges,
                     &edge_properties](vertex_t v_src) {
                      for (std::size_t e = 0; e < num_observations; ++e) {
                        edges.emplace_back(edge_pair_t(v_src, vertex_counter));
                        edge_properties.emplace_back(e);
                      }
                      ++vertex_counter;
                    });
    }

    // insert edges from each source to a target
    if (!trg.empty()) {
      auto v_trg = trg.front();
      for (std::size_t e = 0; e < num_observations; ++e) {
        std::for_each(src.begin(), src.end(),
                      [v_trg, e, &edges, &edge_properties](vertex_t v_src) {
                        edges.emplace_back(edge_pair_t(v_src, v_trg));
                        edge_properties.emplace_back(e);
                      });
      }
    }
  }

  return PolicyGraph(edges.begin(), edges.end(), edge_properties.begin(),
                     vertex_counter, edges.size(), properties);
}

unsigned int steps_remaining(vertex_t v, const PolicyGraph& g) {
  if (boost::out_degree(v, g) == 0)
    return 0;
  else {
    const edge_t e = *(boost::out_edges(v, g).first);
    return (1 + steps_remaining(boost::target(e, g), g));
  }
}

vertex_t find_root(const PolicyGraph& g) {
  vertex_t root(0);
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    if (steps_remaining(root, g) < steps_remaining(v, g)) root = v;
  }
  return root;
}

std::pair<typename boost::filter_iterator<
              vertex_predicate_t,
              typename boost::graph_traits<PolicyGraph>::vertex_iterator>,
          typename boost::filter_iterator<
              vertex_predicate_t,
              typename boost::graph_traits<PolicyGraph>::vertex_iterator> >
vertices_with_steps_remaining(unsigned int s, const PolicyGraph& g) {
  vertex_predicate_t hp = [s, &g](vertex_t v) {
    return (steps_remaining(v, g) == s);
  };
  return filter_vertices(hp, g);
}

std::pair<typename boost::filter_iterator<
              vertex_predicate_t,
              typename boost::graph_traits<PolicyGraph>::vertex_iterator>,
          typename boost::filter_iterator<
              vertex_predicate_t,
              typename boost::graph_traits<PolicyGraph>::vertex_iterator> >
possible_successors(vertex_t v, const PolicyGraph& g) {
  const unsigned int s = steps_remaining(v, g);
  vertex_predicate_t hp = [s, &g](vertex_t v_other) {
    return (s == (steps_remaining(v_other, g) + 1));
  };
  return filter_vertices(hp, g);
}

std::size_t num_possible_successors(vertex_t v, const PolicyGraph& g) {
  auto s = possible_successors(v, g);
  return std::distance(s.first, s.second);
}

std::size_t num_out_edge_configurations(vertex_t v, const PolicyGraph& g) {
  const std::size_t s = steps_remaining(v, g);
  if (s <= 1)
    return 1;
  else
    return std::pow(num_possible_successors(v, g),
                    g[boost::graph_bundle].num_observations_);
}

std::size_t num_local_policies(vertex_t v, const PolicyGraph& g) {
  return g[boost::graph_bundle].num_actions_ *
         num_out_edge_configurations(v, g);
}

bool same_policy(vertex_t v, vertex_t u, const PolicyGraph& g) {
  if (v == u) return true;
  if (g[v] != g[u]) return false;  // different action

  if ((g[v]) == g[u] && (steps_remaining(v, g) == 1) && (steps_remaining(u, g) == 1))
    return true; // last step, and the actions are equal

  // if after same observation, the next actions are different, then the
  // policies are not locally same
  for (auto vep = boost::out_edges(v, g); vep.first != vep.second;
       ++vep.first) {
    for (auto uep = boost::out_edges(u, g); uep.first != uep.second;
         ++uep.first) {
      // check if observation same
      if (g[*vep.first] != g[*uep.first]) continue;
      // check if next nodes' policies  different
      if (!same_policy(boost::target(*vep.first, g),
                       boost::target(*uep.first, g), g))
        return false;
    }
  }
  return true;
}

void set_out_edge_configuration(vertex_t v,
                                std::size_t out_edge_configuration_idx,
                                PolicyGraph& g) {
  if ((steps_remaining(v, g) == 1) || (boost::out_degree(v, g) == 0)) return;

  IndexSpace<std::size_t> I(std::vector<std::size_t>(
      g[boost::graph_bundle].num_observations_, num_possible_successors(v, g)));

  const auto s = possible_successors(v, g);
  boost::clear_out_edges(v, g);
  for (std::size_t e = 0; e < I.local_index_size(); ++e) {
    boost::add_edge(
        v, *std::next(s.first, I.local_index(out_edge_configuration_idx, e)), e,
        g);
  }
}

void set_local_policy(vertex_t v, std::size_t local_policy_idx,
                      PolicyGraph& g) {
  IndexSpace<std::size_t> I(std::vector<std::size_t>{
      g[boost::graph_bundle].num_actions_, num_out_edge_configurations(v, g)});

  g[v] = I.local_index(local_policy_idx, 0);
  set_out_edge_configuration(v, I.local_index(local_policy_idx, 1), g);
}

std::size_t get_local_policy(vertex_t v, std::size_t action_index,
                             std::size_t out_edge_configuration_idx,
                             const PolicyGraph& g) {
  IndexSpace<std::size_t> I(std::vector<std::size_t>{
      g[boost::graph_bundle].num_actions_, num_out_edge_configurations(v, g)});
  return I.joint_index(
      std::vector<std::size_t>{action_index, out_edge_configuration_idx});
}

std::ostream& print(std::ostream& out, const PolicyGraph& g,
                    const std::vector<std::string>& actionnames,
                    const std::vector<std::string>& observationnames) {
  return pgi::print(out, g, VertexWriter<PolicyGraph>(&g, &actionnames),
                    EdgeWriter<PolicyGraph>(&g, &observationnames));
}

std::vector<vertex_t> vertices_with_n_steps(unsigned int n,
                                            const PolicyGraph& g) {
  auto vp = vertices_with_steps_remaining(n, g);
  return std::vector<vertex_t>(vp.first, vp.second);
}

bool any_same_policy(vertex_t v, const std::vector<vertex_t>& v_compare,
                     const PolicyGraph& g) {
  return std::any_of(v_compare.begin(), v_compare.end(), [&g, v](vertex_t vc) {
    return ((v != vc) && same_policy(v, vc, g));
  });
}

bool redirect_in_edges_of_same_policy(vertex_t from,
                                      const std::vector<vertex_t>& to,
                                      PolicyGraph& g, PRNG& rng) {
  bool redirected = false;
  for (auto& q_other : to) {
    if ((q_other != from) && same_policy(q_other, from, g)) {
      redirected = true;
      redirect_in_edges(from, q_other, g);
      randomize_local_policy(
          g, from, rng, vertices_with_n_steps(steps_remaining(from, g), g));
    }
  }
  return redirected;
}

void randomize(PolicyGraph& g, PRNG& rng) {
  for (unsigned int s = 0; s <= g[boost::graph_bundle].num_steps_; ++s) {
    randomize_nodes_with_steps_remaining(s, g, rng);
  }
}

void randomize_nodes_with_steps_remaining(unsigned int steps, PolicyGraph& g,
                                          PRNG& rng) {
  std::vector<vertex_t> avoid_same_policy;
  for (auto qp :
       boost::make_iterator_range(vertices_with_steps_remaining(steps, g))) {
    randomize_local_policy(g, qp, rng, avoid_same_policy);
    avoid_same_policy.push_back(qp);
  }
}

void randomize_local_policy(PolicyGraph& g, vertex_t v, PRNG& rng,
                            const std::vector<vertex_t>& avoid_same_policy) {
  using LocalPolicyDist = boost::random::uniform_int_distribution<std::size_t>;
  LocalPolicyDist d(0, num_local_policies(v, g) - 1);
  do {
    set_local_policy(v, rng(d), g);
  } while (any_same_policy(v, avoid_same_policy, g));
}

}  // namespace pgi
