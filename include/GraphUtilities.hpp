// GraphUtilities.hpp
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

#ifndef GRAPHUTILITIES_HPP
#define GRAPHUTILITIES_HPP
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/graphviz.hpp>
#include <vector>

// all_paths_helper and all_paths modified from original by StackOverflow user
// sehe, https://stackoverflow.com/a/46985434/5471520
namespace pgi {
template <class Graph, class VertexPredicate>
std::pair<
    typename boost::filter_iterator<
        VertexPredicate, typename boost::graph_traits<Graph>::vertex_iterator>,
    typename boost::filter_iterator<
        VertexPredicate, typename boost::graph_traits<Graph>::vertex_iterator> >
filter_vertices(VertexPredicate pred, const Graph& g) {
  auto vit = boost::vertices(g);
  return std::make_pair(
      boost::make_filter_iterator(pred, vit.first, vit.second),
      boost::make_filter_iterator(pred, vit.second, vit.second));
}

template <typename Graph, typename Report>
void all_paths_helper(typename Graph::vertex_descriptor from,
                      typename Graph::vertex_descriptor to, const Graph& g,
                      std::vector<typename Graph::edge_descriptor>& path,
                      const Report& callback) {
  if (from == to) {
    callback(path);
  } else {
    for (auto out : boost::make_iterator_range(boost::out_edges(from, g))) {
      path.push_back(out);
      auto v = boost::target(out, g);
      if (path.end() ==
          std::find_if(path.begin(), path.end(),
                       [&g, v](typename Graph::edge_descriptor e) {
                         return (boost::source(e, g) == v);
                       })) {
        all_paths_helper(v, to, g, path, callback);
      }
      path.pop_back();
    }
  }
}

template <typename Graph, typename Report>
void all_paths(typename Graph::vertex_descriptor from,
               typename Graph::vertex_descriptor to, const Graph& g,
               const Report& callback) {
  std::vector<typename Graph::edge_descriptor> state;
  all_paths_helper(from, to, g, state, callback);
}

template <typename Graph, typename Report>
void all_paths_helper_exhaustive(
    typename Graph::vertex_descriptor v, const Graph& g,
    std::vector<typename Graph::edge_descriptor>& path,
    const Report& callback) {
  callback(path);
  for (auto out : boost::make_iterator_range(boost::out_edges(v, g))) {
    path.push_back(out);
    auto v = boost::target(out, g);
    if (path.end() == std::find_if(path.begin(), path.end(),
                                   [&g, v](typename Graph::edge_descriptor e) {
                                     return (boost::source(e, g) == v);
                                   })) {
      all_paths_helper_exhaustive(v, g, path, callback);
    }
    path.pop_back();
  }
}

template <typename Graph, typename Report>
void all_paths(typename Graph::vertex_descriptor from, const Graph& g,
               const Report& callback) {
  std::vector<typename Graph::edge_descriptor> state;
  all_paths_helper_exhaustive(from, g, state, callback);
}

template <typename Graph>
void redirect_out_edges(typename Graph::vertex_descriptor from,
                        typename Graph::vertex_descriptor to, Graph& g) {
  std::vector<typename boost::edge_bundle_type<Graph>::type> edge_properties;
  for (auto out : boost::make_iterator_range(boost::out_edges(from, g)))
    edge_properties.push_back(g[out]);

  boost::clear_out_edges(from, g);
  for (auto ep : edge_properties) boost::add_edge(from, to, ep, g);
}

template <typename Graph>
void redirect_in_edges(typename Graph::vertex_descriptor from,
                       typename Graph::vertex_descriptor to, Graph& g) {
  typedef std::pair<typename Graph::vertex_descriptor,
                    typename boost::edge_bundle_type<Graph>::type>
      source_edge_pair;
  std::vector<source_edge_pair> svp;

  for (auto in : boost::make_iterator_range(boost::in_edges(from, g)))
    svp.emplace_back(std::make_pair(boost::source(in, g), g[in]));

  boost::clear_in_edges(from, g);
  for (auto& se : svp) boost::add_edge(se.first, to, se.second, g);
}

template <typename Graph>
bool has_path(typename Graph::vertex_descriptor from,
              typename Graph::vertex_descriptor to, const Graph& g) {
  class reachability_visitor : public boost::default_bfs_visitor {
   public:
    reachability_visitor(typename Graph::vertex_descriptor trg) : trg_(trg) {}
    void discover_vertex(typename Graph::vertex_descriptor v,
                         const Graph& g) const {
      ;
      (void)g;  // no-op silences "unused parameter" warning
      if (v == trg_) throw std::runtime_error("Found query node");
    }
    typename Graph::vertex_descriptor trg_;
  };

  reachability_visitor vis(to);
  try {
    boost::breadth_first_search(g, from, boost::visitor(vis));
  } catch (std::runtime_error& e) {
    return true;
  }

  return false;
}

template <typename Graph>
std::pair<typename Graph::out_edge_iterator, bool> out_edge_exists(
    typename Graph::vertex_descriptor v,
    const typename boost::edge_bundle_type<Graph>::type& edge_property,
    const Graph& g) {
  auto eop = boost::out_edges(v, g);
  auto ef =
      std::find_if(eop.first, eop.second,
                   [&g, &edge_property](typename Graph::edge_descriptor e) {
                     return g[e] == edge_property;
                   });
  return std::make_pair(ef, ef != eop.second);
}

template <typename Tree>
std::vector<typename Tree::edge_descriptor> path_from_root_to(
    typename Tree::vertex_descriptor v, const Tree& t) {
  std::vector<typename Tree::edge_descriptor> path;
  while (boost::in_degree(v, t) == 1) {
    typename Tree::edge_descriptor e_in = *(boost::in_edges(v, t).first);
    path.emplace_back(e_in);
    v = boost::source(e_in, t);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

template <typename Graph>
class EdgeWriter {
 public:
  typedef typename boost::graph_traits<Graph>::edge_descriptor Edge;
  EdgeWriter(const Graph* g, const std::vector<std::string>* names)
      : g_(g), names_(names) {}
  void operator()(std::ostream& out, const Edge& e) const {
    auto obs_index = (*g_)[e];
    out << "[label=\"" << obs_index << " ["
        << (names_->empty() ? "" : names_->at(obs_index)) << "]\"]";
  }

 private:
  const Graph* g_;
  const std::vector<std::string>* names_;
};

template <typename Graph>
class VertexWriter {
 public:
  typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
  VertexWriter(const Graph* g, const std::vector<std::string>* names)
      : g_(g), names_(names) {}
  void operator()(std::ostream& out, const Vertex& v) const {
    auto action_index = (*g_)[v];
    auto s = steps_remaining(v, *g_);
    if (s >= 1) {
      out << "[label=\"" << v << " (H" << s << "): " << action_index << " ["
          << (names_->empty() ? "" : names_->at(action_index)) << "]\"]";
    } else {
      out << "[label=\"" << v << " (end)\"]";
    }
  }

 private:
  const Graph* g_;
  const std::vector<std::string>* names_;
};

template <typename Graph>
std::ostream& print(std::ostream& os, const Graph& g,
                    const VertexWriter<Graph>& vpw,
                    const EdgeWriter<Graph>& epw) {
  boost::write_graphviz(os, g, vpw, epw);
  return os;
}

}  // namespace pgi

#endif  // GRAPHUTILITIES_HPP
