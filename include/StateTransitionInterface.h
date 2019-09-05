// StateTransitionInterface.h
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

#ifndef STATETRANSITION_INTERFACE_H
#define STATETRANSITION_INTERFACE_H
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

namespace pgi
{
class StateTransitionInterface
{
   public:
   	virtual ~StateTransitionInterface() {}
	virtual void set(std::size_t next_state, std::size_t state, std::size_t j_act, double p) = 0;
   	virtual double get(std::size_t next_state, std::size_t state, std::size_t j_act) const = 0;
    virtual void predict(Eigen::VectorXd& state_probabilities, std::size_t j_act) const = 0;
    virtual void predict(Eigen::SparseVector<double>& state_probabilities, std::size_t j_act) const = 0;
    virtual Eigen::VectorXd predicted(const Eigen::VectorXd& state_probabilities, std::size_t j_act) const = 0;
    virtual Eigen::SparseVector<double> predicted(const Eigen::SparseVector<double>& state_probabilities, std::size_t j_act) const = 0;
    virtual std::size_t sample_next_state(std::size_t state, std::size_t j_act, double random01) const = 0;
};
}

#endif  // STATETRANSITION_INTERFACE_H
