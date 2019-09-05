// main.cpp
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

#include "BackwardPass.h"
#include "Belief.hpp"
#include "DecPOMDPDiscrete.h"
#include "HistoryCache.hpp"
#include "JointPolicy.h"
#include "JointPolicyUtilities.h"
#include "MADPWrapper.h"
#include "MADPWrapperUtils.h"
#include "PRNG.h"
#include "PolicyInitialization.h"
#include "ValueFunction.h"

#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

void write_policy_to_file(const std::vector<pgi::PolicyGraph>& locals,
                          const std::string& fileprefix,
                          const pgi::DecPOMDPDiscrete& decpomdp) {
  for (std::size_t agent = 0; agent < decpomdp.num_agents(); ++agent) {
    const std::string fn =
        fileprefix + "agent" + std::to_string(agent) + ".dot";
    std::ofstream fs(fn);
    print(fs, locals[agent],
          pgi::element_names(decpomdp.joint_action_space().get(agent)),
          pgi::element_names(decpomdp.joint_observation_space().get(agent)));
  }
}

int main(int argc, char** argv) {
  unsigned int rng_seed;
  unsigned int horizon;
  unsigned int width;
  int improvement_steps;
  std::size_t blind_policy_initial_joint_action;
  std::size_t max_history_cache_size;
  pgi::PolicyInitialization policy_initialization;

  namespace po = boost::program_options;
  po::options_description config("NPGI planner for (Dec)POMDPs \nUsage: " +
                                 std::string(argv[0]) +
                                 " [OPTION]... [DPOMDP-FILE]\nOptions");
  config.add_options()("help", "produce help message")(
      "verbose,v", po::bool_switch()->default_value(false),
      "toggle for verbose output")(
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
      "use-lower-bound,l", po::bool_switch()->default_value(false),
      "toggle to use lower bound")(
      "use-entropy-reward,e", po::bool_switch()->default_value(false),
      "toggle to use expected posterior entropy as final step reward")(
      "use-sparse,t", po::bool_switch()->default_value(false),
      "toggle to use sparse")(
      "seed,s", po::value<unsigned int>(&rng_seed)->default_value(1234567890),
      "random number seed")("output-prefix,o",
                            po::value<std::string>()->default_value("./"),
                            "output prefix")(
      "max-cache-size,m",
      po::value<std::size_t>(&max_history_cache_size)->default_value(5e7),
      "maximum history cache size");

  po::positional_options_description pp;
  pp.add("dpomdp-file", -1);

  po::options_description hidden("Hidden options");
  hidden.add_options()("dpomdp-file", po::value<std::string>(), "input file");

  po::options_description cmdline_options;
  cmdline_options.add(config).add(hidden);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(cmdline_options)
                .positional(pp)
                .run(),
            vm);
  po::notify(vm);
  if (vm.count("help") || argc == 1) {
    std::cout << config << std::endl;
    return 1;
  }
  if (!vm.count("dpomdp-file")) {
    std::cout << "You must specify an input dpomdp file" << std::endl;
    return 1;
  }

  pgi::madpwrapper::MADPDecPOMDPDiscrete madp(
      vm["dpomdp-file"].as<std::string>());
  if (vm["verbose"].as<bool>())
    std::cout << "Loaded problem file " << vm["dpomdp-file"].as<std::string>()
              << std::endl;

  pgi::DecPOMDPDiscrete decpomdp(
      pgi::make_state_space(madp), pgi::make_joint_action_space(madp),
      pgi::make_joint_observation_space(madp),
      pgi::make_transition_matrix(madp, vm["use-sparse"].as<bool>()),
      pgi::make_observation_matrix(madp, vm["use-sparse"].as<bool>()),
      pgi::make_reward_matrix(madp, vm["use-entropy-reward"].as<bool>()));

  pgi::belief_t initial_belief =
      pgi::make_initial_belief(madp, vm["use-sparse"].as<bool>());

  pgi::PRNG rng(rng_seed);

  std::vector<pgi::PolicyGraph> local_policy_graphs;
  const pgi::JointActionSpace& jas = decpomdp.joint_action_space();
  const pgi::JointObservationSpace& jos = decpomdp.joint_observation_space();
  for (std::size_t agent = 0; agent < decpomdp.num_agents(); ++agent) {
    local_policy_graphs.emplace_back(pgi::fixed_width(
        vm["horizon"].as<unsigned int>(), width, jas.num_local_indices(agent),
        jos.num_local_indices(agent)));
  }

  if (policy_initialization == pgi::PolicyInitialization::random) {
    pgi::set_random(local_policy_graphs, rng);
  } else if (policy_initialization == pgi::PolicyInitialization::greedy) {
    pgi::set_open_loop_greedy(local_policy_graphs, initial_belief, decpomdp);
  } else if (policy_initialization == pgi::PolicyInitialization::blind) {
    pgi::set_random(local_policy_graphs,
                    rng);  // to randomize the edges in the policy
    pgi::set_blind(local_policy_graphs, blind_policy_initial_joint_action,
                   decpomdp.joint_action_space());
  }

  if (vm["verbose"].as<bool>())
    std::cout << "Initialized policy using " << policy_initialization
              << " strategy" << std::endl;

  std::ostringstream ss;
  ss << vm["output-prefix"].as<std::string>() << "step" << std::setw(3)
     << std::setfill('0') << 0 << "_";
  write_policy_to_file(local_policy_graphs, ss.str(), decpomdp);

  std::ofstream fvalue(vm["output-prefix"].as<std::string>() +
                       "policy_values.txt");
  std::ofstream ftime(vm["output-prefix"].as<std::string>() +
                      "duration_microseconds.txt");

  pgi::HistoryCache cache(pgi::HistoryData{1.0, 0.0, 0.0, std::move(initial_belief)}, max_history_cache_size);
  pgi::JointPolicy jp(local_policy_graphs);
  double best_value = pgi::value(jp, jp.root(), pgi::History(), decpomdp, cache);
  fvalue << best_value << std::endl;
  std::cout << "Policy value: " << best_value << std::endl;
  std::cout << "Stored " << cache.size() << " histories in cache"
            << std::endl;

  // write initial best policy and value
  for (std::size_t agent = 0; agent < decpomdp.num_agents(); ++agent) {
    std::ofstream fs(vm["output-prefix"].as<std::string>() +
                     "best_policy_agent" + std::to_string(agent) + ".dot");
    print(fs, jp.local_policy(agent),
          pgi::element_names(decpomdp.joint_action_space().get(agent)),
          pgi::element_names(decpomdp.joint_observation_space().get(agent)));
  }

  std::ofstream fs(vm["output-prefix"].as<std::string>() + "best_value.txt");
  fs << best_value << std::endl;

  for (int i = 1; i <= improvement_steps; ++i) {
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    pgi::backpass::BackPassProperties backprops(
        vm["use-lower-bound"].as<bool>());
    auto bp = pgi::backpass::improve(jp, decpomdp, cache, rng, backprops);
    std::chrono::high_resolution_clock::time_point t3 =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t3 - t1).count();
    ftime << duration << std::endl;


    fvalue << bp.improved_policy_value << std::endl;

    if (bp.improved_policy_value > best_value) {
      best_value = bp.improved_policy_value;
      jp = bp.improved_policy;

      // update best policy and value!
      for (std::size_t agent = 0; agent < decpomdp.num_agents(); ++agent) {
        std::ofstream fs(vm["output-prefix"].as<std::string>() +
                         "best_policy_agent" + std::to_string(agent) + ".dot");
        print(fs, jp.local_policy(agent),
              pgi::element_names(decpomdp.joint_action_space().get(agent)),
              pgi::element_names(decpomdp.joint_observation_space().get(agent)));
      }

      std::ofstream fs(vm["output-prefix"].as<std::string>() + "best_value.txt");
      fs << best_value << std::endl;
    }

    std::cout << "Step " << i << " of " << improvement_steps << ": "
              << bp.improved_policy_value << " (best: " << best_value << ")"
              << ", " << duration << " microseconds" << std::endl;

    std::ostringstream ss;
    ss << vm["output-prefix"].as<std::string>() << "/step" << std::setw(3)
       << std::setfill('0') << i << "_";
    write_policy_to_file(jp.local_policies(), ss.str(), decpomdp);

    cache.ensure_size_within_limits();
  }

  return 0;
}
