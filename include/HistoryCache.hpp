// HistoryCache.hpp
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

#ifndef HISTORYCACHE_HPP
#define HISTORYCACHE_HPP
#include <boost/functional/hash.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <unordered_map>
#include "GraphTraversal.hpp"
#include "GraphUtilities.hpp"
#include "HistoryData.h"

namespace pgi {

template <typename K, typename T>
class TreeCache {
 private:
  using vecS = boost::vecS;
  using bidirectionalS = boost::bidirectionalS;
  typedef boost::adjacency_list<vecS, vecS, bidirectionalS, T, K> Tree;

 public:
  typedef typename Tree::vertex_descriptor tree_vertex_t;
  typedef typename Tree::edge_descriptor tree_edge_t;
  typedef typename Tree::out_edge_iterator out_edge_iterator;
  typedef std::vector<tree_edge_t> tree_path_t;
  typedef GraphTraversal<Tree> traversal_t;
  typedef std::vector<K> key_vector_t;
  typedef typename std::vector<K>::const_iterator key_vector_const_iterator;

  TreeCache(T root_data, std::size_t max_size = 1000000)
      : max_size_(max_size),
        t_(),
        root_(boost::add_vertex(std::move(root_data), t_)),
        m_({{key_vector_t(), root_}}) {}

  traversal_t get_traversal(const key_vector_t& key) const
  {
    return GraphTraversal<Tree>(get_path(key), t_);
  }

  const T& get_data(tree_vertex_t v) const { return t_[v]; }

  std::size_t size() const { return m_.size(); }

  std::size_t max_size() const { return max_size_; }

  void ensure_size_within_limits() {
    if (size() > max_size_)
    {
      reset();
    }
  }

  std::pair<tree_vertex_t, key_vector_const_iterator> find_longest_subkey(
      const key_vector_t& key) const {
    key_vector_const_iterator it = key.end();
    auto ip = m_.find(key);
    if (ip == m_.end()) {
      key_vector_t kq(key);
      while (ip == m_.end() && !kq.empty()) {
        kq.pop_back();
        --it;
        ip = m_.find(kq);
      }
    }
    return std::make_pair(ip->second, it);
  }

  std::pair<tree_vertex_t, tree_edge_t> add_vertex(tree_vertex_t parent, const K& key,
                                            const T& data) {
    auto ek = out_edge_exists(parent, key, t_);
    if (ek.second) return std::make_pair(boost::target(*ek.first, t_), *ek.first);

    // vertex does not exist
    tree_vertex_t trg = boost::add_vertex(data, t_);
    std::pair<tree_edge_t, bool> ep = boost::add_edge(parent, trg, key, t_);
    assert(ep.second);

    m_.emplace(std::make_pair(get_key(path_from_root_to(trg, t_)), trg));

    return std::make_pair(trg, ep.first);
  }

 private:
  void reset() {
    const T root_data = get_data(root_);
    t_ = Tree();
    root_ = boost::add_vertex(std::move(root_data), t_);
    m_.clear();
    m_ = key_vertex_map_t({{key_vector_t(), root_}});
  }

  typedef std::unordered_map<key_vector_t, typename Tree::vertex_descriptor,
                             boost::hash<key_vector_t> >
      key_vertex_map_t;

  key_vector_t get_key(const tree_path_t& p) const {
    key_vector_t k;
    k.reserve(p.size());
    std::transform(p.begin(), p.end(), std::back_inserter(k),
                   [this](tree_edge_t e) { return t_[e]; });
    return k;
  }

  tree_path_t get_path(const key_vector_t& k) const
  {
    auto ip = m_.find(k);
    if (ip == m_.end()) throw std::runtime_error("TreeCache::get_path could not find key");
    return path_from_root_to(ip->second, t_);
  }

  std::size_t max_size_;
  Tree t_;
  tree_vertex_t root_;
  key_vertex_map_t m_;

};

typedef TreeCache<ActionObservation, HistoryData> HistoryCache;

}  // namespace pgi
#endif  // HISTORYCACHE_HPP
