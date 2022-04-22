import random
from typing import Any, Dict, List, Tuple
import gym
import numpy as np
import copy

# VNF_TYPE = 0
COST = 0
VNF_TOLERANCE = 1
VNF_BANDWIDTH = 2
VNF_REQUIRE = 3
PLACED = 4
# HOST_TYPE = 0
HOST_CAPACITY = 0
HOST_LINK = 1
REMAINING_CAPACITY = 2
REMAINING_LINK = 3

# VNF_TYPE_MIN = 1
# VNF_TYPE_MAX = 5
COST_MIN = -5
COST_MAX = 0
VNF_TOLERANCE_MIN = 10
VNF_TOLERANCE_MAX = 40
PLACEMENT_MIN = -1

# HOST_TYPE_MIN = 1
HOST_CAPACITY_MIN = 10 * 2
VNF_REQUIRE_MIN = 1
HOST_LINK_MIN = 10 * 2
VNF_BANDWIDTH_MIN = 10
# HOST_TYPE_MAX = 40
HOST_CAPACITY_MAX = 40 * 2
VNF_REQUIRE_MAX = 2
HOST_LINK_MAX = 40 * 4
VNF_BANDWIDTH_MAX = 40

LATENCY_WEIGHT = 1
NUM_VNFS_USED_WEIGHT = 10

class Vnf_Env(gym.Env):

    #mandatory
    def __init__(self, num_vnfs: int = 30, num_hosts: int = 10) -> None:
        PLACEMENT_MAX = num_hosts

        self.num_vnfs = num_vnfs
        self.num_hosts = num_hosts

        self.gen_vnfs, self.gen_hosts = self._get_state()

        self.curr_step = 0
        self.curr_episode = 0
        self.reward = 0

        self.action_space = gym.spaces.Discrete(num_hosts)
        # self.action_space = gym.spaces.Discrete(num_hosts + 1)
        self.ACTION_SPACE_SIZE = self.num_hosts
        # self.ACTION_SPACE_SIZE = self.num_hosts + 1

        low = self.flatten([self.num_vnfs, self.flatten([[COST_MIN for _ in range(self.num_hosts)],
            VNF_TOLERANCE_MIN, VNF_BANDWIDTH_MIN, VNF_REQUIRE_MIN, PLACEMENT_MIN] * self.num_vnfs)])
        self.vnf_length = len(low)
        low2 = [HOST_CAPACITY_MIN, HOST_LINK_MIN] * self.num_hosts
        self.host_length = len(low2)

        high = self.flatten([self.num_vnfs, self.flatten([[COST_MAX for _ in range(self.num_hosts)], 
            VNF_TOLERANCE_MAX, VNF_BANDWIDTH_MAX, VNF_REQUIRE_MAX, PLACEMENT_MAX] * self.num_vnfs) ])
        high2 = [HOST_CAPACITY_MAX, HOST_LINK_MAX] * self.num_hosts
        self.low = np.array(np.concatenate((low, low2)))
        self.high = np.array(np.concatenate((high, high2)))
        
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.OBSERVATION_SPACE_SIZE = len(self.low)

    def generate_vnf(self):
        cost = [random.uniform(COST_MIN, COST_MAX) for _ in range(self.num_hosts)]
        vnf_tolerance = random.uniform(VNF_TOLERANCE_MIN, VNF_TOLERANCE_MAX)
        vnf_bandwidth = random.uniform(VNF_BANDWIDTH_MIN, VNF_BANDWIDTH_MAX)
        vnf_require = random.randint(VNF_REQUIRE_MIN, VNF_REQUIRE_MAX)
        placed = -1
        return np.array([cost, vnf_tolerance, vnf_bandwidth, vnf_require, placed], dtype=object)

    def generate_host(self):
        host_capacity = random.uniform(HOST_CAPACITY_MIN, HOST_CAPACITY_MAX)
        host_link = random.uniform(HOST_LINK_MIN, HOST_LINK_MAX)
        return np.array([host_capacity, host_link])

    def _generate_vnfs(self):
        return [self.generate_vnf() for _ in range(self.num_vnfs)]

    def flatten(self, matrix):
        return np.hstack(matrix)

    def _generate_hosts(self):
        return [self.generate_host() for _ in range(self.num_hosts)]

    #State will be a feature vector
    def _get_state(self):
        return self._generate_vnfs(), self._generate_hosts()

    #State will be a feature vector
    def _set_state(self, vnfs, hosts):
        self.vnfs = vnfs
        self.hosts = hosts
        return self.vnfs, self.hosts

    def get_state_vector(self, vnfs, hosts, step=None):
        if step is None:
            step = self.curr_step
        ct = [x[0] / HOST_CAPACITY_MAX for x in hosts]
        wt = [x[1] / HOST_LINK_MAX for x in hosts]
        vt = self.flatten(vnfs[step][:-1])
        for i in range(self.num_hosts):
            vt[i] /= -COST_MIN
        vt[-3] /= VNF_TOLERANCE_MAX
        vt[-2] /= VNF_BANDWIDTH_MAX
        vt[-1] /= VNF_REQUIRE_MAX
        first = np.concatenate((ct, wt))
        return np.concatenate((first, vt))

    # def get_reward(self, state=None):
    #     if state is None:
    #         vnfs, hosts = self.vnfs, self.hosts
    #     else:
    #         vnfs, hosts = state

    #     # reward = 0
    #     vnf_placed = vnfs[self.curr_step]
    #     action = vnf_placed[PLACED]
    #     for host in hosts:
    #         #constraints
    #         if host[0] < 0:
    #             self.reward -= 10000
    #         if host[-1] < 0:
    #             self.reward -= 10000

    #     if action == self.ACTION_SPACE_SIZE - 1:
    #         self.reward -= 100
    #     else:
    #         self.reward -= vnf_placed[COST][action]

    #     return self.reward

    def get_reward(self, state=None):
        if state is None:
            vnfs, hosts = self.vnfs, self.hosts
        else:
            vnfs, hosts = state

        vnf_placed = vnfs[self.curr_step]
        action = vnf_placed[PLACED]
        self.reward += vnf_placed[COST][action]


        if hosts[action][0] < 0:
            self.reward -= 100

        if hosts[action][1] < 0:
            self.reward -= 100

        return self.reward

    # def get_reward(self, state=None):
    #     if state is None:
    #         vnfs, hosts = self.vnfs, self.hosts
    #     else:
    #         vnfs, hosts = state

    #     if self.curr_step == self.num_vnfs - 1:

    #         costs = np.array([x[COST] for x in vnfs])
    #         decision_variables = np.array([x[PLACED] for x in vnfs])

    #         #Everytime you violate a constraint you should be punished
    #         hard_constraint_violations = 0

    #         #The first objective is to minimize latency
    #         latency = 0

    #         #The second objective is to maximize the number of vnfs used
    #         num_vnfs_used = np.sum(decision_variables.reshape(-1))


    #         for i in range(self.num_vnfs):
    #             latency_host = 0
    #             for j in range(self.num_hosts):
    #                 if decision_variables[i] == j:
    #                     latency_host += costs[i,j]

    #                 # #CONSTRAINT 5
    #                 # if vnfs[i][VNF_TYPE]*decision_variables[i][j] > hosts[j][HOST_TYPE]:
    #                 #     hard_constraint_violations -= 100

    #             #CONSTRAINT 1
    #             if latency_host > vnfs[i][VNF_TOLERANCE]:
    #                 hard_constraint_violations -= 10000

    #             latency -= latency_host

                    
    #         for i in range(self.num_hosts):
    #             bandwidth_vnf = 0
    #             capacity_required = 0
    #             for j in range(self.num_vnfs):
    #                 if decision_variables[j] == i:
    #                     bandwidth_vnf += vnfs[j][VNF_REQUIRE]
    #                     capacity_required += vnfs[j][VNF_BANDWIDTH]

    #             #Constraint 2
    #             if bandwidth_vnf > hosts[i][HOST_CAPACITY]:
    #                 hard_constraint_violations -= 10000
    #             #Constraint 3
    #             if capacity_required > hosts[i][HOST_LINK]:
    #                 hard_constraint_violations -= 10000

    #             latency -= latency_host


    #         return LATENCY_WEIGHT * latency + NUM_VNFS_USED_WEIGHT * num_vnfs_used + hard_constraint_violations
    #     else:
    #         return 0

    #mandatory
    def step(self, action: int) -> Tuple[List[int], float, bool, Dict[Any, Any]]:
        self._take_action(action)
        reward = self.get_reward()
        done = self.check_if_done()
        self.curr_step += 1
        return (self.vnfs, self.hosts), reward, done, {}

    def _take_action(self, action: int) -> None:
        #No placement
        self.vnfs[self.curr_step][PLACED] = action
        if action == self.num_hosts:
            return None
        vnf = self.vnfs[self.curr_step]
        #decrease hosts capacity by vnf_require
        self.hosts[action][HOST_CAPACITY] -= vnf[VNF_REQUIRE]
        #decrease hosts link by vnf_bandwidth
        self.hosts[action][HOST_LINK] -= vnf[VNF_BANDWIDTH]
        return None

    #mandatory
    #restarts the environment and returns initial state
    def reset(self) -> List[int]:
        self.reward = 0
        self.curr_step = 0
        self.curr_episode += 1
        self.vnfs, self.hosts = self._get_state()
        # self.vnfs = copy.deepcopy(self.gen_vnfs)
        # self.hosts = copy.deepcopy(self.gen_hosts)
        return self.vnfs, self.hosts

    #mandatory
    def _render(self, mode: str = "human", close: bool = False) -> None:
        print(self.vnfs)
        print(self.hosts)
        return None

    def check_if_done(self):
        return self.curr_step >= self.num_vnfs - 1

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed

    def increment_step(self):
        self.curr_step += 1

    def invalid_placement(self):
        self.reward -= 10