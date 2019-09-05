# Non-linear policy graph improvement
Non-linear policy graph improvement algorithm for finding policies for decentralized partially observable Markov decision processes, or Dec-POMDPs.

## Decentralized POMDPs
A Dec-POMDP is a decentralized decision-making problem under uncertainty, where a team of agents acts jointly to maximize the expected sum of shared rewards.
The objective is to find for each agent a policy that tells how that agent should act to maximize the shared reward.

The algorithm implemented here is suitable for standard Dec-POMDPs and ones that have an information-theoretic reward function, that is, the reward depends on the joint information state of the agents.
Such Dec-POMDPs can model decentralized information gathering tasks, such as monitoring or surveillance tasks.

The algorithm represents policies as finite state controllers, or policy graphs.
The policy graphs are iteratively improved by finding the action labels of nodes, and the configuration of the out-edges corresponding to observations.

## Publications
For more background information on the algorithm and Dec-POMDPs with information-theoretic rewards, you can refer to the paper:

Mikko Lauri, Joni Pajarinen, Jan Peters. *Information gathering in decentralized POMDPs by policy graph improvement*, in 18th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2019)

You can find a [preprint on arXiv](https://arxiv.org/abs/1902.09840).

BiBTeX:
```
@inproceedings{Lauri2019information,
  author    = {Mikko Lauri and Joni Pajarinen and Jan Peters}, 
  title     = {Information gathering in decentralized POMDPs by policy graph improvement},
  booktitle = {Autonomous Agents and Multiagent Systems (AAMAS)},
  year    = 2019
}
```

# Compilation and installation
Your compiler must support C++14.

Required system libraries:
- eigen3
- Boost libraries: program_options regex graph

Required external libraries:
- [MADP toolbox](https://github.com/MADPToolbox/MADP) v.0.4.1 will be installed automatically, or you can provide a path where to find the library if you have it already by `cmake` options `-DMADPPATH=/path/to/madp`


## How to compile
```
mkdir build && cd build
cmake ..
make
```

## Recommended extra software
You can use the `xdot` viewer tool to visualize the policy graphs.

# Getting started

## Problem domains
The MAV and Information gathering rovers domains from the AAMAS paper are included in the subfolder `problems`.

## Solving a Dec-POMDP
After you have compiled the software, you can try to find a policy for the classical Tiger Dec-POMDP.
We can try with a random seed equal to 100 with the option `-s` and run 100 improvement steps on the policy graphs by the `-i` option:
```
./NPGI -s 100 -i 100 ../problems/dectiger.dpomdp
```

You will see some outputs with policy values and timings.
The software will create some files in the current directory:
- `best_policy_agentI.dot` the best policy for agent `I` in GraphViz DOT format
- `best_value.txt` the value of the best joint policy found by applying best policy graphs for each agent
- `stepXYZ_agentI.dot` the policy graph of agent `I` after improvement step `XYZ` in the GraphViz DOT Format
- `policy_values.txt` the values of policies found at each improvement step
- `duration_microseconds.txt` timing for how long each improvement step took in microseconds


## Solving a Dec-POMDP with information-theoretic rewards
The software supports using negative Shannon information entropy as the final step reward.
This can be useful for information gathering tasks such as those discussed in the paper.
If you want to solve a Dec-POMDP that uses negative entropy as the reward on the final step, just run `NPGI` with the option `-e`.

For example, we can solve the MAV problem from the AAMAS paper for a horizon of 3 steps, with policy graph width of 2.
First, remember to first uncompress the problem file in the `problems` folder!
Then run:
```
./NPGI -e -h 3 -w 2 -i 30 ../problems/mav.dpomdp
```

The outputs are as in the usual Dec-POMDP case.

### Modifying the reward
The software only supports using the negative entropy of the belief state as a final reward.
If you want to modify the information reward, you can do so by modifying the two functions named `final_reward` in the file `RewardMatrix.hpp`.

### Solving a continuous-state Dec-POMDP
You can look at the files `solve_graphsensing.cpp`, `GraphSensingProblem.h` and `GraphSensingProblem.cpp` to see how you can define your own problem with a continuous state.

# Authors
* Mikko Lauri - University of Hamburg, Hamburg, Germany

# License
Licensed under the Apache 2.0 license - see [LICENSE](LICENSE) for details.

# Acknowledgments
- Joni Pajarinen for providing a reference implementation of the original policy graph improvement algorithm
- We use the parser from the excellent [MADP toolbox](https://github.com/MADPToolbox/MADP) to read .dpomdp files.
- We use [combinations_cpp](https://github.com/artem-ogre/combinations_cpp) by Artem Amirkhanov to enumerate combinations.
