// SIRFilter.hpp
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

#ifndef SIRFILTER_HPP
#define SIRFILTER_HPP
#include "History.h"
#include "PRNG.h"
#include "ParticleUtilities.h"
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>

namespace pgi {
template <typename State, typename StateTransitionModel,
          typename ObservationModel>
void SIR(std::vector<State>& states, std::vector<double>& weights,
         const History& h, const StateTransitionModel& t,
         const ObservationModel& o, PRNG& rng) {
  assert(states.size() == weights.size());
  for (const auto& ao : h) {
    SIR_step(states, weights, ao.action_index_, ao.observation_index_, t, o,
             0.2 * static_cast<double>(weights.size()), rng);
  }
}

template <typename State, typename StateTransitionModel,
          typename ObservationModel>
double SIR_step(std::vector<State>& states, std::vector<double>& weights,
              std::size_t j_act, std::size_t j_obs,
              const StateTransitionModel& t, const ObservationModel& o,
              double effective_sample_size_threshold, PRNG& rng) {
  double po = 0.0; // p(o | a)
  for (std::size_t i = 0; i < states.size(); ++i) {
    states[i] = t.sample_next_state(states[i], j_act, rng);
    const double p = o.get(j_obs, states[i], j_act);
    po += weights[i] * p;
    weights[i] *= p;
  }
  normalize(weights);
  if (effective_sample_size_threshold > effective_size(weights)) {
    resample_particles(states, weights, rng);
    for (auto& s : states) {
      s.reinvigorate(rng);
    }
  }
  return po;
}

template <typename State>
void sample_to_fixed_size(std::vector<State>& states,
  std::vector<double>& weights, int N, PRNG& rng)
{
  std::size_t sz = weights.size();
  const int ns = N - static_cast<int>(sz);
  if (ns == 0)
    return;

  if (ns < 0)
  {
    // need to discard samples
    std::vector<State> keep_states(N);
    std::vector<double> keep_weights(N);
    for (int i = 0; i < N; ++i)
    {
      boost::random::discrete_distribution<> dist(weights.begin(), weights.end());
      const auto k = rng(dist);
      keep_states[i] = states[k];
      keep_weights[i] = weights[k];
      weights[k] = 0.0;
    }
    states = keep_states;
    weights = keep_weights;
    normalize(weights);
  }
  else
  {
    // need more samples
    resample_particles(states, weights, rng);

    weights.resize(N);
    std::fill(weights.begin(), weights.end(), 1.0 / static_cast<double>(N));

    states.resize(N);

    boost::random::uniform_int_distribution<std::size_t> dist(0, weights.size()-1);
    for (int i = 0; i < ns; ++i)
    {
      const std::size_t r = rng(dist);
      states[sz+i] = states[r];
      states[sz+i].reinvigorate(rng);
    }
  }
}

template <typename State>
void resample_particles(std::vector<State>& states,
                        std::vector<double>& weights, PRNG& rng) {
  std::vector<std::size_t> resample_indices(0, weights.size());
  resample(weights, resample_indices, rng);

  std::vector<State> resampled_states(states);
  for (std::size_t i = 0; i < states.size(); ++i)
    resampled_states[i] = states[resample_indices[i]];
  states = resampled_states;
  std::fill(weights.begin(), weights.end(),
            1.0 / static_cast<double>(weights.size()));
}

}  // namespace pgi

#endif  // SIRFILTER_HPP
