// ForwardPassParticle.hpp
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

#ifndef FORWARDPASSPARTICLE_HPP
#define FORWARDPASSPARTICLE_HPP
#include "JointPolicy.h"
#include "ObservationModel.hpp"
#include "PRNG.h"
#include "Particle.hpp"
#include "SIRFilter.hpp"
#include "StateTransitionModel.hpp"

namespace pgi {
template <typename State>
class ForwardPassParticle {
 public:
  template <typename StateTransitionModel, typename ObservationModel>
  ForwardPassParticle(ParticleSet<State> particles, const JointPolicy& jp,
                      const StateTransitionModel& t, const ObservationModel& o,
                      const JointActionSpace& jas,
                      const JointObservationSpace& jos, PRNG& rng)
      : m_{{jp.root(), particles}} {
    static_assert(std::is_base_of<pgi::StateTransitionModel<State>,
                                  StateTransitionModel>::value,
                  "StateTransitionModel must inherit from "
                  "pgi::StateTransitionModel<State>");
    static_assert(
        std::is_base_of<pgi::ObservationModel<State>, ObservationModel>::value,
        "ObservationModel must inherit from "
        "pgi::ObservationModel<State>");
    for (unsigned int s = jp.max_steps() - 1; s >= jp.min_steps(); --s) {
      particles = step_forward(particles, jp, t, o, jas, jos, rng);

      for (const auto& q : jp.joint_vertices_with_steps_remaining(s)) {
        auto qp = get_particles_at_node(particles, q);
        m_.insert(std::make_pair(q, qp));
      }
    }
  }

  const ParticleSet<State>& particles_at(JointPolicy::joint_vertex_t q) const {
    return m_.at(q);
  }

 private:
  typedef std::map<JointPolicy::joint_vertex_t, ParticleSet<State> >
      belief_map_t;
  belief_map_t m_;
};
}  // namespace pgi
#endif  // FORWARDPASSAPARTICLE_HPP
