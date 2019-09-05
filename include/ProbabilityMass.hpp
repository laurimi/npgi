// ProbabilityMass.hpp
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

#ifndef PROBABILITY_MASS_HPP
#define PROBABILITY_MASS_HPP
#include <memory>
#include "EigenUtils.hpp"
#include "ObservationInterface.h"
#include "RewardInterface.h"
#include "StateTransitionInterface.h"

namespace pgi {

template <typename T>
typename T::Scalar entropy(const T& x) {
  return detail::entropy(x);
}

template <typename T>
typename T::Index sample(const T& x, typename T::Scalar random01) {
  return detail::sample_from_pmf(x, random01);
}

template <typename T>
typename T::Index sample_space_size(const T& x) {
  return x.size();
}

template <typename Scalar, typename Index = Eigen::Index>
class probability_mass_t {
 public:
  template <typename T>
  probability_mass_t(T x)
      : self_(std::make_shared<pmf_model<T>>(std::move(x))) {}

  friend Scalar entropy(const probability_mass_t& x) {
    return x.self_->entropy_();
  }

  friend Index sample_state(const probability_mass_t& x, Scalar random01) {
    return x.self_->sample_(random01);
  }

  friend Index sample_space_size(const probability_mass_t& x) {
    return x.self_->sample_space_size_();
  }

  friend Scalar reward(const probability_mass_t& x, const RewardInterface& r,
                       std::size_t j_act) {
    return x.self_->reward_(r, j_act);
  }

  friend Scalar final_reward(const probability_mass_t& x,
                             const RewardInterface& r) {
    return x.self_->final_reward_(r);
  }

  friend probability_mass_t successor(const probability_mass_t& x,
                                      const StateTransitionInterface& t,
                                      const ObservationInterface& o,
                                      std::size_t j_act, std::size_t j_obs,
                                      Scalar& p_obs) {
    return x.self_->successor_(t, o, j_act, j_obs, p_obs);
  }

  friend probability_mass_t predicted(const probability_mass_t& x,
                                      const StateTransitionInterface& t,
                                      std::size_t j_act) {
    return x.self_->predicted_(t, j_act);
  }

  friend Eigen::Matrix<Scalar, Eigen::Dynamic, 1> as_vector(
      const probability_mass_t& x) {
    return x.self_->as_vector_();
  }

 private:
  struct pmf_concept_t {
    virtual ~pmf_concept_t() = default;
    virtual Scalar entropy_() const = 0;
    virtual Index sample_(Scalar random01) const = 0;
    virtual Index sample_space_size_() const = 0;
    virtual Scalar reward_(const RewardInterface& r,
                           std::size_t j_act) const = 0;
    virtual Scalar final_reward_(const RewardInterface& r) const = 0;
    virtual pmf_concept_t* successor_(const StateTransitionInterface& t,
                                      const ObservationInterface& o,
                                      std::size_t j_act, std::size_t j_obs,
                                      Scalar& p_obs) const = 0;
    virtual pmf_concept_t* predicted_(const StateTransitionInterface& t,
                                      std::size_t j_act) const = 0;
    virtual Eigen::Matrix<Scalar, Eigen::Dynamic, 1> as_vector_() const = 0;
  };

  template <typename T>
  struct pmf_model final : pmf_concept_t {
    pmf_model(T x) : data_(std::move(x)) {}

    Scalar entropy_() const override { return entropy(data_); }

    Index sample_(Scalar random01) const override {
      return sample(data_, random01);
    }

    Index sample_space_size_() const override {
      return sample_space_size(data_);
    }

    Scalar reward_(const RewardInterface& r, std::size_t j_act) const override {
      return r.reward(data_, j_act);
    }

    Scalar final_reward_(const RewardInterface& r) const override {
      return r.final_reward(data_);
    }

    pmf_concept_t* successor_(const StateTransitionInterface& t,
                              const ObservationInterface& o, std::size_t j_act,
                              std::size_t j_obs, Scalar& p_obs) const override {
      T predicted_data_ = t.predicted(data_, j_act);
      p_obs = o.update(predicted_data_, j_act, j_obs);
      return new pmf_model(predicted_data_);
    }

    pmf_concept_t* predicted_(const StateTransitionInterface& t,
                              std::size_t j_act) const override {
      return new pmf_model(t.predicted(data_, j_act));
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> as_vector_() const override {
      return Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(data_);
    }

    T data_;
  };

  probability_mass_t(pmf_concept_t* x) : self_(x) {}

  std::shared_ptr<const pmf_concept_t> self_;
};
}  // namespace pgi

#endif  // PROBABILITY_MASS_HPP
