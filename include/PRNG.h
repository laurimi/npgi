// PRNG.h
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

#ifndef PRNG_H
#define PRNG_H
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>

namespace pgi
{
class PRNG
{
   public:
    typedef boost::random::mt19937 Engine;

    // If no seed is specified, use current time
    PRNG(unsigned int seed) : engine_(Engine(seed)) {}
    // Call by giving instance of a distribution, for example of type:
    // boost::random::normal_distribution<double>
    template <class DistributionType>
    typename DistributionType::result_type operator()(DistributionType& d)
    {
        return d(engine_);
    }

   private:
    Engine engine_;
};

template <typename T>
bool is_rnd01_below(T threshold, PRNG& rng) {
  boost::random::uniform_01<T> d;
  return (rng.operator()(d) < threshold);
}

}  // namespace pgi
#endif  // PRNG_H
