// solve_graphsensing.cpp
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

#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "BackwardPassParticle.hpp"
#include "GraphSensingProblem.h"
#include "JointPolicyUtilities.h"
#include "Particle.hpp"
#include "PolicyInitialization.h"

int main(int argc, char** argv) {
  unsigned int rng_seed;
  unsigned int horizon;
  unsigned int width;
  int improvement_steps;
  int num_particles_fwd;
  int num_rollouts;
  int num_particle_rollout;
  std::size_t blind_policy_initial_joint_action;
  pgi::PolicyInitialization policy_initialization;

  double mx, my, sx, sy;  // optional gaussian initial belief params

  namespace po = boost::program_options;
  po::options_description config("NPGI planner for (Dec)POMDPs \nUsage: " +
                                 std::string(argv[0]) +
                                 " [OPTION]...\nOptions");
  config.add_options()("help", "produce help message")(
      "horizon,h", po::value<unsigned int>(&horizon)->default_value(3),
      "problem horizon")("policy-width,w",
                         po::value<unsigned int>(&width)->default_value(3),
                         "policy width")(
      "policy-initialization,p",
      po::value<pgi::PolicyInitialization>(&policy_initialization)
          ->default_value(pgi::PolicyInitialization::random),
      "policy initialization type: (random | greedy | blind)")(
      "blind-action,b",
      po::value<std::size_t>(&blind_policy_initial_joint_action)
          ->default_value(0),
      "action for blind policy initialization")(
      "improvement-steps,i",
      po::value<int>(&improvement_steps)->default_value(9),
      "number of policy improvement steps")(
      "num-rollouts,r",
      po::value<int>(&num_rollouts)->default_value(100),
      "number of rollouts for value estimation")
      (
      "num-particles-rollout,k", po::value<int>(&num_particle_rollout)->default_value(100),
      "number of particles per rollout")(
      "num-particles,n", po::value<int>(&num_particles_fwd)->default_value(1000),
      "number of particles in P(s,q) forward pass")(
      "seed,s", po::value<unsigned int>(&rng_seed)->default_value(1234567890),
      "random number seed")(
      "output-prefix,o", po::value<std::string>()->default_value("./"),
      "output prefix")("gaussian", po::bool_switch()->default_value(false),
                       "toggle for Gaussian initial belief")(
      "mx", po::value<double>(&mx)->default_value(0.1),
      "initial belief mean x value")("my",
                                     po::value<double>(&my)->default_value(0.1),
                                     "initial belief mean y value")(
      "sx", po::value<double>(&sx)->default_value(0.1),
      "initial belief x value standard deviation")(
      "sy", po::value<double>(&sy)->default_value(0.1),
      "initial belief y value standard deviation");

  po::options_description cmdline_options;
  cmdline_options.add(config);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(),
            vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << config << std::endl;
    return 1;
  }

  pgi::PRNG rng(rng_seed);

  // Create the GraphSensing Dec-POMDP problem
  pgi::JointActionSpace jas = pgi::GraphSensing::joint_action_space;
  pgi::GraphSensing::StateTransitionModel t;
  pgi::GraphSensing::RewardModel r;

  // RSS
  pgi::JointObservationSpace jos =
      pgi::GraphSensing::rss_joint_observation_space;
  pgi::GraphSensing::RSSObservationModel o;

  // Initialize policy
  std::vector<pgi::PolicyGraph> local_policy_graphs;
  for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent) {
    local_policy_graphs.emplace_back(pgi::fixed_width(
        vm["horizon"].as<unsigned int>(), width, jas.num_local_indices(agent),
        jos.num_local_indices(agent)));
  }

  if (policy_initialization == pgi::PolicyInitialization::random) {
    pgi::set_random(local_policy_graphs, rng);
  } else if (policy_initialization == pgi::PolicyInitialization::greedy) {
    throw std::runtime_error(
        "Greedy open loop initialization not implemented for continuous-state "
        "problems!");
  } else if (policy_initialization == pgi::PolicyInitialization::blind) {
    pgi::set_random(local_policy_graphs,
                    rng);  // to randomize the edges in the policy
    pgi::set_blind(local_policy_graphs, blind_policy_initial_joint_action, jas);
  }
  pgi::JointPolicy jp(local_policy_graphs);

  // Create initial belief
  pgi::ParticleSet<pgi::GraphSensing::state_t> init_particles;
  if (vm["gaussian"].as<bool>()) {
    pgi::GraphSensing::sample_initial_states_gaussian(
        init_particles.states_, num_particles_fwd,
        pgi::GraphSensing::location_t{mx, my}, sx, sy, rng);
  } else {
    pgi::GraphSensing::sample_initial_states(init_particles.states_,
                                             num_particles_fwd, rng);
  }
  init_particles.nodes_ =
      std::vector<pgi::JointPolicy::joint_vertex_t>(num_particles_fwd, jp.root());
  init_particles.weights_ = std::vector<double>(
      num_particles_fwd, 1.0 / static_cast<double>(num_particles_fwd));

  // Set up output streams
  std::ofstream fvalue(vm["output-prefix"].as<std::string>() +
                       "policy_values.txt");
  std::ofstream ftime(vm["output-prefix"].as<std::string>() +
                      "duration_microseconds.txt");

  // Get value of initial policy
  double best_value = estimate_value(num_rollouts, init_particles.states_,
                                     init_particles.weights_, jp, jp.root(), t,
                                     o, r, jas, jos, rng);

  fvalue << best_value << std::endl;
  std::cout << "Policy value: " << best_value << std::endl;

  // write initial best policy and value
  for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent) {
    std::ofstream fs(vm["output-prefix"].as<std::string>() +
                     "best_policy_agent" + std::to_string(agent) + ".dot");
    print(fs, jp.local_policy(agent), pgi::element_names(jas.get(agent)),
          pgi::element_names(jos.get(agent)));
  }

  std::ofstream fs(vm["output-prefix"].as<std::string>() + "best_value.txt");
  fs << best_value << std::endl;

  // Backward pass setup
  pgi::backpass::BackPassProperties props;
  props.prob_random_history_in_heuristic_improvement = 0.05;

  for (int i = 1; i <= improvement_steps; ++i) {

    // sample new set of particles for improvement
    if (vm["gaussian"].as<bool>()) {
      pgi::GraphSensing::sample_initial_states_gaussian(
          init_particles.states_, num_particles_fwd,
          pgi::GraphSensing::location_t{mx, my}, sx, sy, rng);
    } else {
      pgi::GraphSensing::sample_initial_states(init_particles.states_,
                                               num_particles_fwd, rng);
    }
    init_particles.nodes_ =
        std::vector<pgi::JointPolicy::joint_vertex_t>(num_particles_fwd, jp.root());
    init_particles.weights_ = std::vector<double>(
        num_particles_fwd, 1.0 / static_cast<double>(num_particles_fwd));

    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    auto bp = pgi::backpass::improve_particle(num_rollouts, num_particle_rollout, init_particles, jp, t, o, r, jas,
                                              jos, rng, props);
    std::chrono::high_resolution_clock::time_point t3 =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t1).count();
    ftime << duration << std::endl;

    // check
    double improved_policy_value = estimate_value(
        num_rollouts, init_particles.states_, init_particles.weights_, bp.improved_policy,
        bp.improved_policy.root(), t, o, r, jas, jos, rng);

    fvalue << improved_policy_value << std::endl;

    if (improved_policy_value > best_value) {
      best_value = improved_policy_value;
      jp = bp.improved_policy;

      // Update best value and policy!
      for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent) {
        std::ofstream fs(vm["output-prefix"].as<std::string>() +
                         "best_policy_agent" + std::to_string(agent) + ".dot");
        print(fs, jp.local_policy(agent), pgi::element_names(jas.get(agent)),
              pgi::element_names(jos.get(agent)));
      }

      std::ofstream fs(vm["output-prefix"].as<std::string>() + "best_value.txt");
      fs << best_value << std::endl;
    }

    std::cout << "Step " << i << " of " << improvement_steps << ": "
              << improved_policy_value << " (best: " << best_value << ")"
              << ", " << duration << " microseconds" << std::endl;

    std::ostringstream ss;
    ss << vm["output-prefix"].as<std::string>() << "/step" << std::setw(3)
       << std::setfill('0') << i << "_";

    for (std::size_t agent = 0; agent < jas.num_local_spaces(); ++agent) {
      std::ofstream fs(ss.str() + "agent" + std::to_string(agent) + ".dot");
      print(fs, bp.improved_policy.local_policy(agent),
            pgi::element_names(jas.get(agent)),
            pgi::element_names(jos.get(agent)));
    }
  }

  pgi::ForwardPassParticle<pgi::GraphSensing::state_t> fwd(init_particles, jp,
                                                           t, o, jas, jos, rng);
  std::cout << "Forward pass results: \n";
  for (unsigned int s = 1; s <= jp.max_steps(); ++s) {
  for (auto& q : jp.joint_vertices_with_steps_remaining(s))
  {
    auto ql = jp.to_local(q);
    for (auto& qq : ql)
      std::cout << qq << " ";

    const pgi::ParticleSet<pgi::GraphSensing::state_t>& p = fwd.particles_at(q);
    std::cout << ": " << p.weights_.size() << " particles\n";

  }
  }

  return 0;
}
