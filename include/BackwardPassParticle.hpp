// BackwardPassParticle.hpp
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

#ifndef BACKWARDPASSPARTICLE_HPP
#define BACKWARDPASSPARTICLE_HPP
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include "BackwardPass.h"
#include "ForwardPassParticle.hpp"
#include "GraphSensingProblem.h"
#include "JointPolicyHistories.h"
#include "Particle.hpp"
#include "SIRFilter.hpp"

namespace pgi {
namespace backpass {
template <typename State, typename StateTransitionModel,
          typename ObservationModel, typename RewardModel>
ImprovementResult improve_particle(int num_rollouts, int num_particle_rollout,
                                   const ParticleSet<State>& init_particles,
                                   JointPolicy jp,
                                   const StateTransitionModel& t,
                                   const ObservationModel& o,
                                   const RewardModel& r,
                                   const JointActionSpace& jas,
                                   const JointObservationSpace& jos, PRNG& rng,
                                   const BackPassProperties& props) {
  static_assert(std::is_base_of<pgi::StateTransitionModel<State>,
                                StateTransitionModel>::value,
                "StateTransitionModel must inherit from "
                "pgi::StateTransitionModel<State>");
  static_assert(
      std::is_base_of<pgi::ObservationModel<State>, ObservationModel>::value,
      "ObservationModel must inherit from "
      "pgi::ObservationModel<State>");
  static_assert(std::is_base_of<pgi::RewardModel<State>, RewardModel>::value,
                "RewardModel must inherit from "
                "pgi::RewardModel<State>");

  pgi::ForwardPassParticle<State> fwd(init_particles, jp, t, o, jas, jos, rng);
  double policy_value = 0.0;
  for (std::size_t steps = jp.min_steps(); steps <= jp.max_steps(); ++steps) {
    for (std::size_t agent = 0; agent < jp.num_agents(); ++agent) {
      PolicyGraph& local = jp.local_policy(agent);
      std::vector<vertex_t> already_improved;
      for (const auto qlocal : boost::make_iterator_range(
               vertices_with_steps_remaining(steps, local))) {
        value_map_t vm = estimate_local_policy_values(
            num_rollouts, num_particle_rollout, jp, agent, qlocal, fwd, t, o, r,
            jas, jos, rng, props);

        auto imax = std::max_element(
            vm.begin(), vm.end(), [](const std::pair<std::size_t, stats_t>& a,
                                     const std::pair<std::size_t, stats_t>& b) {
              return boost::accumulators::weighted_sum(a.second) <
                     boost::accumulators::weighted_sum(b.second);
            });
        policy_value = boost::accumulators::weighted_sum(imax->second);
        set_local_policy(qlocal, imax->first, local);
        if (redirect_in_edges_of_same_policy(qlocal, already_improved, local,
                                             rng)) {
          // we must update the expected beliefs if we redirected
          // some edges
          fwd = pgi::ForwardPassParticle<State>(init_particles, jp, t, o, jas,
                                                jos, rng);
        }
        already_improved.push_back(qlocal);
      }
    }
  }

  return ImprovementResult{jp, policy_value};
}

template <typename State, typename StateTransitionModel,
          typename ObservationModel, typename RewardModel>
value_map_t local_policy_values_regular(
    int num_rollouts, int num_particle_rollout, JointPolicy jp,
    const std::size_t agent_idx, const vertex_t agent_vertex,
    const pgi::ForwardPassParticle<State>& fwd, const StateTransitionModel& t,
    const ObservationModel& o, const RewardModel& r,
    const JointActionSpace& jas, const JointObservationSpace& jos, PRNG& rng) {
  value_map_t vm;
  PolicyGraph& local = jp.local_policy(agent_idx);
  for (const auto& q :
       jp.joint_vertices_with_agent_at(agent_idx, agent_vertex)) {
    ParticleSet<State> qp = fwd.particles_at(q);
    if (qp.weights_.empty()) continue;
    const double prob = normalize(qp.weights_);

    if (is_almost_zero(prob)) continue;


    // create required number of particles for rollout
    std::vector<double> w_rollout(qp.weights_);
    std::vector<State> s_rollout(qp.states_);
    sample_to_fixed_size(s_rollout, w_rollout, num_particle_rollout, rng);

   
    for (std::size_t a = 0; a < local[boost::graph_bundle].num_actions_; ++a) {
      local[agent_vertex] = a;
      for (std::size_t e = 0;
           e < num_out_edge_configurations(agent_vertex, local); ++e) {
        set_out_edge_configuration(agent_vertex, e, local);
       
        double value = estimate_value(num_rollouts, s_rollout, w_rollout, jp, q,
                                      t, o, r, jas, jos, rng);
        const std::size_t i = get_local_policy(agent_vertex, a, e, local);
        vm[i](value, boost::accumulators::weight = prob);
      }
    }
  }
  return vm;
}

template <typename State, typename StateTransitionModel,
          typename ObservationModel, typename RewardModel>
value_map_t local_policy_values_heuristic(
    int num_rollouts, int num_particle_rollout, JointPolicy jp,
    const std::size_t agent_idx, const vertex_t agent_vertex,
    const pgi::ForwardPassParticle<State>& fwd, const StateTransitionModel& t,
    const ObservationModel& o, const RewardModel& r,
    const JointActionSpace& jas, const JointObservationSpace& jos, PRNG& rng) {
  value_map_t vm;
  PolicyGraph& local = jp.local_policy(agent_idx);

  // create required number of particles for rollout
  std::vector<double> w_rollout(num_particle_rollout, 1.0/static_cast<double>(num_particle_rollout));
  std::vector<State> s_rollout(num_particle_rollout);
  GraphSensing::sample_initial_states(s_rollout, num_particle_rollout, rng);

  // now draw a random history and update belief
  State s = GraphSensing::sample_initial_state(rng);
  std::pair<History, bool> hs = sample_random_history(s, jp.root(), jp, t, o, jas, rng);
  SIR(s_rollout, w_rollout, hs.first, t, o, rng);

  // pick a random joint policy node where agent is at agent_vertex
  auto qvec = jp.joint_vertices_with_agent_at(agent_idx, agent_vertex);
  boost::random::uniform_int_distribution<> unif(0, qvec.size()-1);
  const JointPolicy::joint_vertex_t q = qvec[rng(unif)];

  for (std::size_t a = 0; a < local[boost::graph_bundle].num_actions_; ++a) {
    local[agent_vertex] = a;
    for (std::size_t e = 0;
         e < num_out_edge_configurations(agent_vertex, local); ++e) {
      set_out_edge_configuration(agent_vertex, e, local);

      double value = estimate_value(num_rollouts, s_rollout, w_rollout, jp, q,
                                    t, o, r, jas, jos, rng);
      const std::size_t i = get_local_policy(agent_vertex, a, e, local);
      vm[i](value, boost::accumulators::weight = 1.0);
    }
  }

  return vm;
}

template <typename State, typename StateTransitionModel,
          typename ObservationModel, typename RewardModel>
value_map_t estimate_local_policy_values(
    int num_rollouts, int num_particle_rollout, const JointPolicy& jp,
    const std::size_t agent_idx, const vertex_t agent_vertex,
    const pgi::ForwardPassParticle<State>& fwd, const StateTransitionModel& t,
    const ObservationModel& o, const RewardModel& r,
    const JointActionSpace& jas, const JointObservationSpace& jos, PRNG& rng,
    const BackPassProperties& props) {

  value_map_t local_policy_values = local_policy_values_regular(
      num_rollouts, num_particle_rollout, jp, agent_idx, agent_vertex, fwd, t,
      o, r, jas, jos, rng);
  if (local_policy_values.empty()) {
    // probably this node is uncreachable, do a heuristic update in it instead
    local_policy_values = local_policy_values_heuristic(
        num_rollouts, num_particle_rollout, jp, agent_idx, agent_vertex, fwd, t,
        o, r, jas, jos, rng);

  }

  return local_policy_values;
}

}  // namespace backpass
}  // namespace pgi
#endif  // BACKWARDPASSPARTICLE_HPP
