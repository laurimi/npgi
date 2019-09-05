// DiscreteAbstractions.hpp
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

#ifndef DISCRETEABSTRACTIONS_H
#define DISCRETEABSTRACTIONS_H
#include <boost/flyweight.hpp>
#include <string>
#include <vector>
#include "IndexSpace.hpp"
namespace pgi {

class NamedUnit {
 public:
  virtual ~NamedUnit() = default;
  NamedUnit(std::string name) : name_(std::move(name)) {}
  const std::string& name() const { return name_.get(); }

 private:
  boost::flyweight<std::string> name_;
};

class DiscreteState : public NamedUnit {
 public:
  DiscreteState(const std::string& name = "unknown") : NamedUnit(name) {}
};

class DiscreteAction : public NamedUnit {
 public:
  DiscreteAction(const std::string& name = "unknown") : NamedUnit(name) {}
};

class DiscreteObservation : public NamedUnit {
 public:
  DiscreteObservation(const std::string& name = "unknown") : NamedUnit(name) {}
};

template <typename T>
class DiscreteSpace {
 public:
  DiscreteSpace(std::vector<T> v) : v_(std::move(v)) {}
  std::size_t size() const { return v_.size(); }
  const T& get(std::size_t index) const { return v_[index]; }

 private:
  std::vector<T> v_;
};

template <typename T>
class JointDiscreteSpace : public IndexSpace<std::size_t> {
 public:
  JointDiscreteSpace(std::vector<DiscreteSpace<T>> local)
      : IndexSpace<std::size_t>([&]{
          std::vector<std::size_t> sz;
          for (const auto& d : local) sz.emplace_back(d.size());
          return sz;
        }()),
        local_(std::move(local)) {}

  std::size_t num_local_spaces() const { return local_.size(); }
  const DiscreteSpace<T>& get(std::size_t index) const { return local_[index]; }

 private:
  std::vector<DiscreteSpace<T>> local_;
};

typedef DiscreteSpace<DiscreteState> StateSpace;
typedef DiscreteSpace<DiscreteAction> ActionSpace;
typedef DiscreteSpace<DiscreteObservation> ObservationSpace;
typedef JointDiscreteSpace<DiscreteAction> JointActionSpace;
typedef JointDiscreteSpace<DiscreteObservation> JointObservationSpace;

template <typename DiscreteSpaceType>
std::vector<std::string> element_names(const DiscreteSpaceType& d)
{
  std::vector<std::string> names;
  for (std::size_t i = 0; i < d.size(); ++i)
    names.emplace_back( d.get(i).name() );
  return names;
}

}  // namespace pgi

#endif  // DISCRETEABSTRACTIONS_H
