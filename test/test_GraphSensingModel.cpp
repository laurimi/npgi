// test_GraphSensingModel.cpp
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

#include <boost/random/uniform_01.hpp>
#include "BackwardPass.h"
#include "BackwardPassParticle.hpp"
#include "ForwardPassParticle.hpp"
#include "GraphSensingProblem.h"
#include "JointPolicy.h"
#include "JointPolicyUtilities.h"
#include "Particle.hpp"

#include "gtest/gtest.h"

TEST(MovementTest, StayInPlace) {
  pgi::GraphSensing::MovementModel m;
  for (int i = 0; i < 6; ++i) {
    ASSERT_EQ(m.next_location(i, i), i);
    ASSERT_TRUE(m.action_allowed_in(i, i));
  }
}

// TEST(MovementTest, AllowedActions) {
//   pgi::GraphSensing::MovementModel m;
//   // 0
//   const std::vector<int> allowed{0, 2};
//   for (const auto& x : allowed) ASSERT_TRUE(m.action_allowed_in(0, x));

//   // 3
//   const std::vector<int> allowed2{2, 3, 4, 5};
//   for (const auto& x : allowed2) ASSERT_TRUE(m.action_allowed_in(3, x));
// }


// TEST(MovementTest, NewLocationsExpected) {
//   pgi::GraphSensing::MovementModel m;

//   ASSERT_EQ(m.next_location(0, 2), 2);
//   ASSERT_EQ(m.next_location(3, 4), 4);
//   ASSERT_EQ(m.next_location(5, 3), 3);
//   ASSERT_EQ(m.next_location(3, 2), 2);
// }

TEST(SIR, Filter)
{
  pgi::JointActionSpace jas = pgi::GraphSensing::joint_action_space;
  pgi::JointObservationSpace jos = pgi::GraphSensing::rss_joint_observation_space;
  pgi::GraphSensing::StateTransitionModel t;
  pgi::GraphSensing::RSSObservationModel o;
  pgi::GraphSensing::RewardModel r;

  pgi::PRNG rng(1234567890);


  const unsigned int width = 1;
  const unsigned int horizon = 1;

  std::vector<pgi::PolicyGraph> local_policy_graphs;
  for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent) {
    local_policy_graphs.emplace_back(
        pgi::fixed_width(horizon, width, jas.num_local_indices(agent),
                         jos.num_local_indices(agent)));
  }
  const std::size_t blind_action = 0;
  pgi::set_blind(local_policy_graphs, blind_action, jas);
  pgi::JointPolicy jp(local_policy_graphs);

  const int num_particles = 100;
  pgi::ParticleSet<pgi::GraphSensing::state_t> p;
  pgi::GraphSensing::sample_initial_states(p.states_, num_particles, rng);
  p.weights_.resize(num_particles);
  std::fill(p.weights_.begin(), p.weights_.end(), 1.0 / static_cast<double>(num_particles));
  p.nodes_.resize(num_particles);
  std::fill(p.nodes_.begin(), p.nodes_.end(), jp.root());


  // std::cout << "Particles with weights: \n";
  // std::cout << "[";
  // for (std::size_t i = 0; i < num_particles; ++i)
  // {
  //   if (i > 0)
  //     std::cout << ",\n";
  //   std::cout << "[" << p.weights_[i] << ", " << p.states_[i].source_location_.x << ", " << p.states_[i].source_location_.y << "]";
  // }
  // std::cout << "]\n";

  // for (std::size_t j_obs = 0; j_obs < jos.num_joint_indices(); ++j_obs)//const std::size_t j_obs = 6;//jos.num_joint_indices()-1;
  // {
  //   std::cout << "j_obs = " << j_obs << "\n";
  //     for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent)
  //     {
  //       const std::size_t obs_local = jos.local_index(j_obs, agent);
  //       std::cout << "agent " << agent << " obs_local= " << obs_local << " - ";
  //       std::cout << jos.get(agent).get(obs_local).name() << "\n";
  //     }}

  // {
  //   std::vector<pgi::GraphSensing::state_t> s(p.states_);
  //   std::vector<double> w(p.weights_);
  //   const std::size_t j_obs = 0;
  //   SIR_step(s, w, blind_action, j_obs, t, o, 0.0, rng);

  //   std::cout << "Particles with weights: \n";
  //   std::cout << "[";
  //   for (std::size_t i = 0; i < num_particles; ++i) {
  //     if (i > 0) std::cout << ",\n";
  //     std::cout << "[" << w[i] << ", "
  //               << s[i].source_location_.x << ", "
  //               << s[i].source_location_.y << "]";
  //   }
  //   std::cout << "]\n";
  // }
  // std::cout << "\n***** \n";

  // {
  //   std::vector<pgi::GraphSensing::state_t> s(p.states_);
  //   std::vector<double> w(p.weights_);
  //   const std::size_t j_obs = 4;
  //   SIR_step(s, w, blind_action, j_obs, t, o, 0.0, rng);

  //   std::cout << "Particles with weights: \n";
  //   std::cout << "[";
  //   for (std::size_t i = 0; i < num_particles; ++i) {
  //     if (i > 0) std::cout << ",\n";
  //     std::cout << "[" << w[i] << ", "
  //               << s[i].source_location_.x << ", "
  //               << s[i].source_location_.y << "]";
  //   }
  //   std::cout << "]\n";
  // }
  // std::cout << "\n***** \n";

  // {
  //   std::vector<pgi::GraphSensing::state_t> s(p.states_);
  //   std::vector<double> w(p.weights_);
  //   const std::size_t j_obs = 8;
  //   SIR_step(s, w, blind_action, j_obs, t, o, 0.0, rng);

  //   std::cout << "Particles with weights: \n";
  //   std::cout << "[";
  //   for (std::size_t i = 0; i < num_particles; ++i) {
  //     if (i > 0) std::cout << ",\n";
  //     std::cout << "[" << w[i] << ", "
  //               << s[i].source_location_.x << ", "
  //               << s[i].source_location_.y << "]";
  //   }
  //   std::cout << "]\n";
  // }

  // for (double d = 0.2; d < 40.0; d += 0.2) {
  //   pgi::GraphSensing::location_t x{0.0, 0.0};
  //   pgi::GraphSensing::location_t y{0.0, d};
  //   const auto p0 = o.get_observation_prob(x, y);

  //   std::cout << "[" << d << ", " << p0[0] << ", " << p0[1] << ", " << p0[2] << "],\n";
  // }
}

// TEST(DecPOMDPContinuousStateTest, ValueTest) {
//   // pgi::DecPOMDPContinuousState<pgi::GraphSensing::state_t> decpomdp(
//   //     pgi::GraphSensing::joint_action_space,
//   //     pgi::GraphSensing::joint_observation_space,
//   //     pgi::GraphSensing::StateTransitionModel(),
//   //     pgi::GraphSensing::ObservationModel(),
//   //     pgi::GraphSensing::RewardModel());

//   // pgi::GraphSensing::RewardModel m = pgi::GraphSensing::RewardModel();

//   pgi::JointActionSpace jas = pgi::GraphSensing::joint_action_space;
//   pgi::JointObservationSpace jos = pgi::GraphSensing::bernoulli_joint_observation_space;
//   pgi::GraphSensing::StateTransitionModel t;
//   pgi::GraphSensing::BernoulliObservationModel o;
//   pgi::GraphSensing::RewardModel r;

//   pgi::PRNG rng(1234567890);

//   const unsigned int width = 2;
//   const unsigned int horizon = 4;

//   std::vector<pgi::PolicyGraph> local_policy_graphs;
//   // const pgi::JointActionSpace& jas = decpomdp.joint_action_space();
//   // const pgi::JointObservationSpace& jos = decpomdp.joint_observation_space();
//   for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent) {
//     local_policy_graphs.emplace_back(
//         pgi::fixed_width(horizon, width, jas.num_local_indices(agent),
//                          jos.num_local_indices(agent)));
//   }

//   // const std::size_t blind_policy_initial_joint_action = 0;
//   // pgi::set_blind(local_policy_graphs, blind_policy_initial_joint_action,
//   // jas);

//   // for (std::size_t j = 0; j < 100; ++j) {
//   pgi::set_random(local_policy_graphs, rng);

//   pgi::JointPolicy jp(local_policy_graphs);

//   pgi::ParticleSet<pgi::GraphSensing::state_t> init_particles;
//   const int N = 100;
//   for (std::size_t i = 0; i < N; ++i) {
//     // draw a random state
//     boost::random::uniform_01<double> ud;
//     pgi::GraphSensing::state_t x;
//     x.agent_0_location_ = 0;
//     x.agent_1_location_ = 0;
//     x.source_location_.x = rng(ud);
//     x.source_location_.y = rng(ud);

//     init_particles.states_.push_back(x);
//     init_particles.weights_.push_back(1.0 / static_cast<double>(N));
//     init_particles.nodes_.push_back(jp.root());
//   }

//   std::cout << "Final reward " << r.final_reward(init_particles.states_, init_particles.weights_) << "\n";

//   //   std::cout << "expected reward: " <<
//   //   expected_sum_of_rewards(init_particles,
//   //                                                               jp, t, o, r,
//   //                                                               jas, jos,
//   //                                                               rng)
//   //             << "\n";
//   // }
// }

// TEST(ForwardPassTest, SimpleTest) {
//   pgi::JointActionSpace jas = pgi::GraphSensing::joint_action_space;
//   pgi::JointObservationSpace jos = pgi::GraphSensing::bernoulli_joint_observation_space;
//   pgi::GraphSensing::StateTransitionModel t;
//   pgi::GraphSensing::BernoulliObservationModel o;
//   pgi::GraphSensing::RewardModel r;

//   pgi::PRNG rng(1234567890);

//   const unsigned int width = 2;
//   const unsigned int horizon = 6;

//   std::vector<pgi::PolicyGraph> local_policy_graphs;
//   for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent) {
//     local_policy_graphs.emplace_back(
//         pgi::fixed_width(horizon, width, jas.num_local_indices(agent),
//                          jos.num_local_indices(agent)));
//   }

//   pgi::set_random(local_policy_graphs, rng);

//   pgi::JointPolicy jp(local_policy_graphs);

//   pgi::ParticleSet<pgi::GraphSensing::state_t> init_particles;
//   const int N = 2000;
//   for (std::size_t i = 0; i < N; ++i) {
//     // draw a random state
//     boost::random::uniform_01<double> ud;
//     pgi::GraphSensing::state_t x;
//     x.agent_0_location_ = 0;
//     x.agent_1_location_ = 5;
//     x.source_location_.x = rng(ud);
//     x.source_location_.y = rng(ud);

//     init_particles.states_.push_back(x);
//     init_particles.weights_.push_back(1.0 / static_cast<double>(N));
//     init_particles.nodes_.push_back(jp.root());
//   }

//   // pgi::ForwardPassParticle<pgi::GraphSensing::state_t> fwd(init_particles, jp,
//   //                                                          t, o, jas, jos, rng);

//   // for (unsigned int s = jp.max_steps(); s >= jp.min_steps(); --s) {
//   //   std::cout << "*** steps remaining: " << s << "****\n";
//   //   for (const auto& q : jp.joint_vertices_with_steps_remaining(s)) {
//   //     const auto& p = fwd.particles_at(q);
//   //     std::cout << "Node " << q << " number of particles " << p.weights_.size()
//   //               << "\n";
//   //   }
//   // }

//   // Backward pass
//   pgi::backpass::BackPassProperties props;

//   double policy_value =
//       pgi::expected_sum_of_rewards(init_particles.states_, init_particles.weights_, jp, jp.root(), t, o, r, jas, jos, rng);
//   for (std::size_t j = 0; j < 30; ++j) {
//     auto bp = pgi::backpass::improve_particle(100, init_particles, jp, t, o, r, jas,
//                                               jos, rng, props);
//     std::cout << "Improved policy value: " << bp.improved_policy_value << "\n";
//     if (policy_value < bp.improved_policy_value) {
//       std::cout << "Improves on old (" << policy_value << "), updating\n";
//       policy_value = bp.improved_policy_value;
//       jp = bp.improved_policy;
//     }
//   }

//   std::cout << "Storing best policy\n";
//   for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent) {
//     std::ofstream fs("best_policy_agent" + std::to_string(agent) + ".dot");
//     print(fs, jp.local_policy(agent), pgi::element_names(jas.get(agent)),
//           pgi::element_names(jos.get(agent)));
//   }
// }

// TEST(ObservationModelTest, ZeroEntropy) {
//   pgi::GraphSensing::BernoulliObservationModel gm;

//   pgi::GraphSensing::state_t x;
//   x.agent_0_location_ = 0;
//   x.agent_1_location_ = 0;
//   x.source_location_.x = 0.2;
//   x.source_location_.y = 0.4;

//   double p_tot = 0.0;
//   for (std::size_t j_obs = 0; j_obs < 4; ++j_obs) {
//     const double pj = gm.get(j_obs, x, 0);
//     p_tot += pj;
//     std::cout << pj << "\n";
//   }

//   EXPECT_NEAR(p_tot, 1.0, 1e-12);
//   std::cout << "total " << p_tot << "\n";

//   pgi::PRNG rng(1234567890);
//   for (std::size_t i = 0; i < 100; ++i)
//     std::cout << gm.sample_observation(x, 0, rng) << " ";
//   std::cout << "\n";
// }
