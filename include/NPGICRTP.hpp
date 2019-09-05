// NPGICRTP.hpp
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

#ifndef NPGICRTP_HPP
#define NPGICRTP_HPP
namespace pgi {
template <typename T>
struct crtp {
  T& underlying() { return static_cast<T&>(*this); }
  T const& underlying() const { return static_cast<T const&>(*this); }
};
}

#endif  // NPGICRTP_HPP
