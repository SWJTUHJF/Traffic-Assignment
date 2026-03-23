import re
from math import inf

import numpy as np


class Node:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.upstream_links: list[Link] = list()
        self.downstream_links: list[Link] = list()
        self.parent: Node = self
        self.dist: float = inf
        self.visited: bool = False  # For SPFA algorithm

    def __repr__(self):
        return f'NODE {self.node_id}'


class Link:
    def __init__(
            self,
            link_id: int,
            tail: Node = None,
            head: Node = None,
            capacity: float = None,
            length: float = None,
            free_flow_time: float = None,
            alpha: float = None,
            beta: float = None):
        self.link_id = link_id
        self.tail = tail
        self.head = head
        self.capacity = capacity
        self.length = length
        self.fft = free_flow_time
        self.alpha = alpha
        self.beta = beta
        self.flow = 0
        self.auxiliary_flow: float = 0  # For link-based
        self.cost: float = self.fft
        self.marginal_cost: float = self.fft

    def __repr__(self):
        return f'LINK {self.link_id} cost = {self.cost}, flow = {self.flow}'

    def update_cost(self) -> None:
        self.cost = self.fft * (1 + self.alpha * (self.flow / self.capacity) ** self.beta)

    def get_specific_cost(self, flow: float) -> float:
        return self.fft * (1 + self.alpha * (flow / self.capacity) ** self.beta)

    def update_marginal_cost(self) -> None:
        self.marginal_cost = (self.fft * (1 + self.alpha * (self.flow / self.capacity) ** self.beta)
                             + self.fft * self.alpha * self.beta * (self.flow / self.capacity) ** self.beta)

    def get_specific_marginal_cost(self, flow) -> float:
        return (self.fft * (1 + self.alpha * (flow / self.capacity) ** self.beta)
                + self.fft * self.alpha * self.beta * (flow / self.capacity) ** self.beta)

    def cost_function_derivative(self) -> float:
        return self.fft * self.alpha * self.beta * self.flow ** (self.beta - 1) / self.capacity ** self.beta

    def marginal_cost_function_derivative(self) -> float:
        return (self.alpha * self.fft * self.flow ** (self.beta - 1) * self.beta * (1 + self.beta) /
                self.capacity ** self.beta)


class ODPair:
    def __init__(
            self,
            origin: Node,
            destination: Node,
            demand: float
    ):
        self.origin = origin
        self.destination = destination
        self.demand = demand
        self.working_path_set: list[Path] = list()

    def __repr__(self):
        return f'ODPair {self.origin.node_id}->{self.destination.node_id}={self.demand}'


class Path:
    def __init__(
            self,
            origin: Node,
            destination: Node,
            included_links: list[Link]
    ):
        self.origin = origin
        self.destination = destination
        self.included_links = included_links
        self.flow: float = 0
        self.cost: float = 0
        self.marginal_cost: float = 0
        self.update_cost()
        self.update_marginal_cost()

    def __repr__(self):
        nodes = [link.tail.node_id for link in self.included_links] + [self.destination.node_id]
        return f"{'-'.join(map(str, nodes))}(cost={self.cost}, flow={self.flow})"

    def __eq__(self, other):
        this_link_sequence = [link.link_id for link in self.included_links]
        other_link_sequence = [link.link_id for link in other.included_links]
        return this_link_sequence == other_link_sequence

    def update_cost(self) -> None:
        self.cost = sum([link.cost for link in self.included_links])

    def update_marginal_cost(self) -> None:
        self.marginal_cost = sum([link.marginal_cost for link in self.included_links])


class Network:
    def __init__(
            self,
            name: str,
            network_root_path: str = "",
            demand_sst: float = 1
    ):
        self.name = name
        self.network_root_path = network_root_path
        self.demand_sst = demand_sst
        self.node_set: list[Node] = list()
        self.link_set: list[Link] = list()
        self.od_pair_set: list[ODPair] = list()
        self.num_node: int = 0
        self.num_link: int = 0
        self.num_od: int = 0
        self.run()

    def run(self):
        self.read_network()
        self.read_OD()

    def read_network(self) -> None:
        if self.network_root_path:
            file_path = f"{self.network_root_path}\\TransportationNetworks\\{self.name}\\{self.name}_net.txt"
        else:
            file_path = f"TransportationNetworks\\{self.name}\\{self.name}_net.txt"
        with open(file_path, 'r', encoding='UTF-8' ) as f1:
            # Process the text file
            pattern = re.compile(r'[\w.~]+')
            lines = f1.readlines()
            data = [pattern.findall(line) for line in lines if len(pattern.findall(line)) != 0]
            self.num_node = int(data[1][-1])
            for i in range(len(data)):
                if '~' in data[i] and "ORIGINAL" not in data[i]:
                    data = data[i + 1:]
                    break
            # Create NODE and LINK object
            self.node_set = [Node(i) for i in range(self.num_node + 1)]  # Be CAREFUL that position 0 represents nothing
            self.link_set = [Link(0)]
            for index, line in enumerate(data):
                temp = Link(index + 1, self.node_set[int(line[0])], self.node_set[int(line[1])], float(line[2]),
                            float(line[3]), float(line[4]), float(line[5]), float(line[6]))
                self.link_set.append(temp)
                self.node_set[int(line[0])].downstream_links.append(temp)
                self.node_set[int(line[1])].upstream_links.append(temp)

    def read_OD(self) -> None:
        if self.network_root_path:
            file_path = f"{self.network_root_path}\\TransportationNetworks\\{self.name}\\{self.name}_trips.txt"
        else:
            file_path = f"TransportationNetworks\\{self.name}\\{self.name}_trips.txt"
        with open(file_path, 'r', encoding='UTF-8') as f1:
            # Process the text file
            pattern = re.compile(r'[0-9.]+|Origin')
            lines = f1.readlines()
            data = [pattern.findall(line) for line in lines if len(pattern.findall(line)) != 0]
            total_flow = float(data[1][0]) * self.demand_sst
            for i in range(len(data)):
                if 'Origin' in data[i]:
                    data = data[i:]
                    break
            # Create OD Pair object
            for line in data:
                if "Origin" in line:
                    origin = self.node_set[int(line[-1])]
                else:
                    for i in range(len(line) // 2):
                        destination, demand = self.node_set[int(line[2 * i])], float(line[2 * i + 1])
                        if demand != 0:
                            self.od_pair_set.append(ODPair(origin, destination, demand * self.demand_sst))
            # Check the correctness of OD flows
            if abs(total_flow - sum([od.demand for od in self.od_pair_set])) > 1:
                raise ValueError("Data in the file does not match with the total OD flow.")

    def get_TSTT(self, verbose: bool = False) -> float:
        total = sum([link.cost * link.flow for link in self.link_set[1:]])
        if verbose:
            print(f'Total system travel time is {total:.2f}')
        return total

    def update_all_link_costs(self) -> None:
        for link in self.link_set[1:]:
            link.update_cost()

    def update_all_link_marginal_costs(self) -> None:
        for link in self.link_set[1:]:
            link.update_marginal_cost()

    def update_all_working_path_costs(self) -> None:
        for od in self.od_pair_set:
            for path in od.working_path_set:
                path.update_cost()

    def update_all_working_path_marginal_costs(self) -> None:
        for od in self.od_pair_set:
            for path in od.working_path_set:
                path.update_marginal_cost()

    def get_flow_vector(self) -> np.ndarray:
        return np.array([link.flow for link in self.link_set[1:]])

    def get_cost_vector(self) -> np.ndarray:
        return np.array([link.cost for link in self.link_set[1:]])

    def get_marginal_cost_vector(self) -> np.ndarray:
        return np.array([link.marginal_cost for link in self.link_set[1:]])

    def reset_flow_and_cost(self):
        for link in self.link_set[1:]:
            link.flow, link.auxiliary_flow = 0, 0
            link.update_cost()
            link.update_marginal_cost()

if __name__ == '__main__':
    sf = Network("SiouxFalls", demand_sst=2)
