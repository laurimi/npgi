// PolicyInitialization.cpp
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

#include "PolicyInitialization.h"
#include <boost/algorithm/string.hpp>

namespace pgi {
std::ostream& operator<<(std::ostream& os, const PolicyInitialization& p) {
  if (p == PolicyInitialization::random)
    os << "RANDOM";
  else if (p == PolicyInitialization::greedy)
    os << "GREEDY";
  else if (p == PolicyInitialization::blind)
    os << "BLIND";
  else if (p == PolicyInitialization::file)
    os << "FILE";
  else
    os << "UNKNOWN";
  return os;
}

std::istream& operator>>(std::istream& in, PolicyInitialization& p) {
  std::string token;
  in >> token;
  boost::to_upper(token);
  if (token == "RANDOM")
    p = PolicyInitialization::random;
  else if (token == "GREEDY")
    p = PolicyInitialization::greedy;
  else if (token == "BLIND")
    p = PolicyInitialization::blind;
  else if (token == "FILE")
    p = PolicyInitialization::file;
  else
    in.setstate(std::ios_base::failbit);
  return in;
}
}  // namespace pgi
