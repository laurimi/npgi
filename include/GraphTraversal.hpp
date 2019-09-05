// GraphTraversal.hpp
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

#ifndef GRAPHTRAVERSAL_HPP
#define GRAPHTRAVERSAL_HPP
#include <boost/graph/adjacency_list.hpp>
#include <string>
#include <vector>
#include "GraphUtilities.hpp"

namespace pgi {

template <class Graph>
class GraphTraversal {
 public:
  typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_t;
  typedef typename boost::graph_traits<Graph>::edge_descriptor edge_t;
  typedef typename boost::graph_traits<Graph>::out_edge_iterator out_edge_iterator;
  typedef std::vector<edge_t> path_t;
  typedef typename boost::vertex_bundle_type<Graph>::type vertex_property_t;
  typedef typename boost::edge_bundle_type<Graph>::type edge_property_t;

  GraphTraversal(vertex_t vstart, const Graph& g)
      : current_vertex_(vstart), path_(), g_(g) {}

  GraphTraversal(path_t path, const Graph& g)
      : current_vertex_(path.empty() ? 0 : boost::target(path.back(), g)),
        path_(std::move(path)),
        g_(g) {}

  vertex_t current_vertex() const { return current_vertex_; }

  const vertex_property_t& current_vertex_properties() const {
    return g_[current_vertex_];
  }

  vertex_t return_to_previous() {
    if (!path_.empty()) {
      current_vertex_ = boost::source(path_.back(), g_);
      path_.pop_back();
    }
    return current_vertex_;
  }

  vertex_t traverse(const edge_property_t& ep) {
    auto ek = out_edge_exists(current_vertex_, ep, g_);
    if (ek.second) {
      return traverse(*ek.first);
    } else {
      throw std::runtime_error("requested out edge not found for vertex " +
                               std::to_string(current_vertex_));
    }
  }

  vertex_t traverse(edge_t e)
  {
    path_.emplace_back(e);
    current_vertex_ = boost::target(e, g_);
    return current_vertex_;
  }

  std::pair<out_edge_iterator, bool> can_traverse(const edge_property_t& ep) const
  {
    return out_edge_exists(current_vertex_, ep, g_);
  }

  bool can_traverse() const {
    return (boost::out_degree(current_vertex_, g_) != 0);
  }

  const path_t& get_path() const { return path_; }

 private:
  vertex_t current_vertex_;
  path_t path_;
  const Graph& g_;
};

}  // namespace pgi

#endif  // GRAPHTRAVERSAL_HPP
