// Particle.hpp
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

#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include <type_traits>
#include "JointPolicy.h"
#include "ObservationModel.hpp"
#include "PRNG.h"
#include "ParticleUtilities.h"
#include "PolicyGraphTraversal.h"
#include "RewardModel.hpp"
#include "StateTransitionModel.hpp"
#include "JointPolicyHistories.h"
#include "common.hpp"
#include <boost/random/discrete_distribution.hpp>

namespace pgi {

template <typename State>
struct ParticleSet {
  std::vector<State> states_;
  std::vector<JointPolicy::joint_vertex_t> nodes_;
  std::vector<double> weights_;
};

template <typename State, typename StateTransitionModel,
          typename ObservationModel>
ParticleSet<State> step_forward(ParticleSet<State> particles,
                                const JointPolicy& jp,
                                const StateTransitionModel& t,
                                const ObservationModel& o,
                                const JointActionSpace& jas,
                                const JointObservationSpace& jos, PRNG& rng) {
  static_assert(std::is_base_of<pgi::StateTransitionModel<State>,
                                StateTransitionModel>::value,
                "StateTransitionModel must inherit from "
                "pgi::StateTransitionModel<State>");
  static_assert(
      std::is_base_of<pgi::ObservationModel<State>, ObservationModel>::value,
      "ObservationModel must inherit from "
      "pgi::ObservationModel<State>");

  for (std::size_t i = 0; i < particles.weights_.size(); ++i) {
    JointPolicyGraphTraversal tv(particles.nodes_[i], jp);
    const std::size_t j_act = tv.current_action(jas);
    const std::size_t j_obs =
        simulate_step(particles.states_[i], j_act, t, o, rng);
    // no weight update needed as observation is also sampled

    if (tv.can_traverse()) {
      tv.traverse(j_obs, jos);
      particles.nodes_[i] = tv.current_vertex();
    } else {
      particles.nodes_[i] = -1;  // TODO: required? better way?
    }
  }
  normalize(particles.weights_);

  return particles;
}

template <typename State, typename StateTransitionModel,
          typename ObservationModel>
std::size_t simulate_step(State& s, std::size_t j_act,
                          const StateTransitionModel& t,
                          const ObservationModel& o, PRNG& rng) {
  s = t.sample_next_state(s, j_act, rng);
  return o.sample_observation(s, j_act, rng);
}

template <typename State, typename StateTransitionModel,
          typename ObservationModel>
std::pair<History, bool> sample_random_history(
    State s, JointPolicy::joint_vertex_t q_start, const JointPolicy& jp,
    const StateTransitionModel& t, const ObservationModel& o,
    const JointActionSpace& jas, PRNG& rng) {
  unsigned int hist_length = jp.max_steps() - jp.steps_remaining(q_start);

  if (hist_length == 0) return std::make_pair(History(), true);

  History h;
  h.reserve(hist_length);
  boost::random::uniform_int_distribution<std::size_t> act_dist(
      0, jas.num_joint_indices() - 1);
  for (unsigned int i = 0; i < hist_length; ++i) {
    const std::size_t j_act = rng(act_dist);
    s = t.sample_next_state(s, j_act, rng);
    std::size_t j_obs = o.sample_observation(s, j_act, rng);
    h.emplace_back(ActionObservation{j_act, j_obs});
  }
  return std::make_pair(h, true);
}

template <typename State, typename StateTransitionModel,
          typename ObservationModel, typename RewardModel>
double estimate_value(int num_rollouts, const std::vector<State>& states,
                      const std::vector<double>& weights, const JointPolicy& jp,
                      JointPolicy::joint_vertex_t q_start,
                      const StateTransitionModel& t, const ObservationModel& o,
                      const RewardModel& r, const JointActionSpace& jas,
                      const JointObservationSpace& jos, PRNG& rng) {
  double avg_value = 0.0;
  for (int n = 0; n < num_rollouts; ++n) {
    const double value =
        rollout(states, weights, jp, q_start, t, o, r, jas, jos, rng);
    avg_value = (value + static_cast<double>(n) * avg_value) /
                static_cast<double>(n + 1);
  }
  return avg_value;
}

template <typename State, typename StateTransitionModel,
          typename ObservationModel, typename RewardModel>
double rollout(std::vector<State> states,
               std::vector<double> weights, const JointPolicy& jp,
               JointPolicy::joint_vertex_t q_start,
               const StateTransitionModel& t, const ObservationModel& o,
               const RewardModel& r, const JointActionSpace& jas,
               const JointObservationSpace& jos, PRNG& rng) {
  double v = 0.0;
  JointPolicyGraphTraversal tv(q_start, jp);
  for (unsigned int s = jp.steps_remaining(q_start); s > 0; --s)
  {
    const std::size_t j_act = tv.current_action(jas);
    v += r.get(states, weights, j_act);

    // sample an observation according to one of the states
    boost::random::discrete_distribution<> dist(weights.begin(), weights.end());
    State s_next = t.sample_next_state(states[rng(dist)], j_act, rng);
    const std::size_t j_obs = o.sample_observation(s_next, j_act, rng);

    if (tv.can_traverse())
    {
      tv.traverse(j_obs, jos);
    }

    // update particles
    SIR_step(states, weights, j_act, j_obs, t, o,
             0.1 * static_cast<double>(weights.size()), rng);
  }

  v += r.final_reward(states, weights);
  return v;
}

template <typename State, typename StateTransitionModel,
          typename ObservationModel, typename RewardModel>
double expected_sum_of_rewards(const std::vector<State>& states,
                               const std::vector<double>& weights,
                               const JointPolicy& jp,
                               JointPolicy::joint_vertex_t q_start,
                               const StateTransitionModel& t,
                               const ObservationModel& o, const RewardModel& r,
                               const JointActionSpace& jas,
                               const JointObservationSpace& jos, PRNG& rng) {
  if (q_start == 1234567890)
  {
    return r.final_reward(states, weights);
  }

  JointPolicyGraphTraversal tv(q_start, jp);
  const std::size_t j_act = tv.current_action(jas);

  // init with immediate reward
  double v = r.get(states, weights, j_act);

  // std::cout << "loop start:\n";
  std::vector<double> v_obs(jos.num_joint_indices(), 0.0);
  std::vector<double> p_obs(jos.num_joint_indices(), 0.0);
  for (std::size_t j_obs = 0; j_obs < jos.num_joint_indices(); ++j_obs)
  {
    std::vector<State> states_obs(states);
    std::vector<double> weights_obs(weights);

    p_obs[j_obs] = SIR_step(states_obs, weights_obs, j_act, j_obs, t, o,
             0.1 * static_cast<double>(weights.size()), rng);
   
    JointPolicyGraphTraversal tv(q_start, jp);
    std::size_t q_next = 1234567890;
    if (jp.steps_remaining(q_start) > 1)
    {
      tv.traverse(j_obs, jos);
      q_next = tv.current_vertex();
    }
    v_obs[j_obs] = expected_sum_of_rewards(states_obs, weights_obs, jp, q_next, t, o,
                                      r, jas, jos, rng);
  }
  normalize(p_obs);
  double v_future = std::inner_product(p_obs.begin(), p_obs.end(), v_obs.begin(), 0.0);

  return v + v_future;
}

template <typename State>
ParticleSet<State> get_particles_at_node(const ParticleSet<State>& particles,
                                         JointPolicy::joint_vertex_t q) {
  ParticleSet<State> qp;
  for (std::size_t i = 0; i < particles.weights_.size(); ++i) {
    if (particles.nodes_[i] == q) {
      qp.states_.push_back(particles.states_[i]);
      qp.nodes_.push_back(particles.nodes_[i]);
      qp.weights_.push_back(particles.weights_[i]);
    }
  }
  return qp;
}
}  // namespace pgi
#endif  // PARTICLE_HPP
