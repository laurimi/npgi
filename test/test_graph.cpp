// test_graph.cpp
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

#include <algorithm>
#include <boost/graph/graphviz.hpp>
#include "GraphUtilities.hpp"
#include "gtest/gtest.h"

class GraphTest : public ::testing::Test {
 protected:
  struct VertexData {
    std::string name;
  };
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                                VertexData>
      Graph;

  typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
  typedef typename boost::graph_traits<Graph>::edge_descriptor Edge;
  typedef std::vector<Edge> Path;

  void SetUp() override {
    const std::string graph_spec("digraph { a -> b; b -> c; c -> d; b -> d;}");
    boost::dynamic_properties dp(boost::ignore_other_properties);
    dp.property("node_id", get(&VertexData::name, g));
    boost::read_graphviz(graph_spec, g, dp);
  }

  Vertex by_vertex_name(const std::string& name) {
    boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
    boost::tie(vi, vi_end) = boost::vertices(g);
    return *std::find_if(vi, vi_end,
                         [&](Vertex v) { return name == g[v].name; });
  };

  // void TearDown() override {}

  Graph g;
};

TEST_F(GraphTest, SameVertexPaths) {
  std::vector<Path> paths;
  pgi::all_paths(by_vertex_name("a"), by_vertex_name("a"), g, [&paths](const Path& p){ paths.push_back(p); });
  ASSERT_EQ(paths.size(), 1);
  EXPECT_EQ(paths.front(), Path());
}

TEST_F(GraphTest, MultiplePaths) {
  std::vector<Path> paths;
  pgi::all_paths(by_vertex_name("a"), by_vertex_name("d"), g, [&paths](const Path& p){ paths.push_back(p); });
  EXPECT_EQ(paths.size(), 2);
}

TEST_F(GraphTest, ZeroPaths) {
  std::vector<Path> paths;
  pgi::all_paths(by_vertex_name("c"), by_vertex_name("a"), g, [&paths](const Path& p){ paths.push_back(p); });
  EXPECT_EQ(paths.size(), 0);
}

TEST_F(GraphTest, ExpectedPathsExist) {
  EXPECT_TRUE(pgi::has_path(by_vertex_name("a"), by_vertex_name("a"), g));
  EXPECT_TRUE(pgi::has_path(by_vertex_name("a"), by_vertex_name("d"), g));
  EXPECT_TRUE(pgi::has_path(by_vertex_name("a"), by_vertex_name("c"), g));
  EXPECT_TRUE(pgi::has_path(by_vertex_name("b"), by_vertex_name("d"), g));
  EXPECT_TRUE(pgi::has_path(by_vertex_name("c"), by_vertex_name("d"), g));
  EXPECT_TRUE(pgi::has_path(by_vertex_name("d"), by_vertex_name("d"), g));
}

TEST_F(GraphTest, ExpectedPathsDoNotExist) {
  EXPECT_FALSE(pgi::has_path(by_vertex_name("d"), by_vertex_name("a"), g));
  EXPECT_FALSE(pgi::has_path(by_vertex_name("c"), by_vertex_name("a"), g));
  EXPECT_FALSE(pgi::has_path(by_vertex_name("b"), by_vertex_name("a"), g));
  EXPECT_FALSE(pgi::has_path(by_vertex_name("c"), by_vertex_name("b"), g));
}

TEST_F(GraphTest, EveryPath) {
  std::vector<Path> paths;
  pgi::all_paths(by_vertex_name("a"), g, [&paths](const Path& p){ paths.push_back(p); });
  EXPECT_EQ(paths.size(), 5);
}

TEST_F(GraphTest, RedirectInEdges) {
  pgi::redirect_in_edges(by_vertex_name("d"), by_vertex_name("a"), g);

  EXPECT_EQ(boost::in_degree(by_vertex_name("d"), g), 0);
  EXPECT_EQ(boost::in_degree(by_vertex_name("a"), g), 2);
}

TEST_F(GraphTest, RedirectOutEdges) {
  pgi::redirect_out_edges(by_vertex_name("a"), by_vertex_name("c"), g);

  EXPECT_EQ(boost::in_degree(by_vertex_name("b"), g), 0);
  EXPECT_EQ(boost::in_degree(by_vertex_name("c"), g), 2);
}
