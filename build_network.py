from __future__ import annotations

from utilities import dijkstra

import re
from typing import Literal


class Node:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.link_in: list[Link] = list()
        self.link_out: list[Link] = list()

    def __repr__(self):
        return f'Node {self.node_id}'


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
            beta: float = None
            ):
        self.link_id = link_id
        self.tail = tail
        self.head = head
        self.capacity = capacity
        self.length = length
        self.fft = free_flow_time
        self.alpha = alpha
        self.beta = beta
        self.flow = 0
        self.aux_flow: float = 0
        self.cost: float = self.fft
        self.marginal_cost: float = self.fft

    def __repr__(self):
        return f'Link {self.link_id}({self.tail}-{self.head}): flow={self.flow:.1f}'

    def update_cost(self) -> None:
        self.cost = self.get_cost(self.flow)

    def get_cost(self, flow: float) -> float:
        return self.fft * (1 + self.alpha * (flow / self.capacity) ** self.beta)

    def update_marginal_cost(self) -> None:
        self.marginal_cost = self.get_marginal_cost(self.flow)

    def get_marginal_cost(self, flow) -> float:
        return self.get_cost(flow) + self.fft * self.alpha * self.beta * (flow / self.capacity) ** self.beta

    def update_cost_and_marginal_cost(self) -> None:
        self.update_cost()
        self.update_marginal_cost()

    @property
    def cost_function_derivative(self) -> float:
        return self.fft * self.alpha * self.beta * self.flow ** (self.beta - 1) / self.capacity ** self.beta

    @property
    def marginal_cost_function_derivative(self) -> float:
        return self.alpha * self.fft * self.flow ** (self.beta - 1) * self.beta * (1 + self.beta) / self.capacity ** self.beta


class OD:
    def __init__(self, origin: Node, destination: Node, demand: float, network: Network):
        self.origin = origin
        self.destination = destination
        self.demand = demand
        self.network = network
        self.working_set: list[Path] = list()

    def __repr__(self):
        return f'OD {self.origin.node_id}->{self.destination.node_id}={self.demand}'
    
    def shortest_path_and_cost(self, cost_type: Literal["c", "mc"] = "c") -> tuple[list[Link], float]:
        path = dijkstra(network=self.network, origin=self.origin, destination=self.destination, cost_type=cost_type)
        path_cost = sum([link.cost for link in path]) if cost_type == "c" else sum([link.marginal_cost for link in path])
        return path, path_cost


class Path:
    def __init__(self, origin: Node, destination: Node, included_links: list[Link]):
        self.origin = origin
        self.destination = destination
        self.included_links = included_links
        self.flow: float = 0

    def __repr__(self):
        nodes = [link.tail.node_id for link in self] + [self.destination.node_id]
        return f"{'-'.join(map(str, nodes))}(cost={self.cost}, flow={self.flow})"
    
    def __iter__(self):
        return iter(self.included_links)

    def __hash__(self):
        return hash(tuple(link.link_id for link in self))

    def __eq__(self, other: Path):
        return self.included_links == other.included_links

    def add_flow(self, value) -> None:
        self.flow += value
        for link in self:
            link.flow += value

    @property
    def cost(self) -> None:
        return sum([link.cost for link in self])

    @property
    def marginal_cost(self) -> None:
        return sum([link.marginal_cost for link in self])


class Network:
    def __init__(self, name: str, demand_level: float = 1, from_file: bool = False):
        self.name = name
        self.demand_level = demand_level
        self.link_set: list[Link] = list()
        self.node_set: list[Node] = list()
        self.od_set: list[OD] = list()
        self.num_node: int = 0
        self.num_link: int = 0
        self.num_od: int = 0
        self.total_flow: float = 0
        self.read_network()
        self.read_OD()
        if from_file:  # Read UE data from file
            self.read_flow()

    def read_network(self) -> None:
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_net.txt", 'r') as f:
            pattern = re.compile(r"[0-9A-Za-z.~]+")
            lines = f.readlines()
            lines = [pattern.findall(line) for line in lines if len(pattern.findall(line)) != 0]
        for i in range(len(lines)):
                line = lines[i]
                if "NUMBER" in line and "NODES" in line:
                    self.num_node = int(line[-1])
                if "NUMBER" in line and "LINKS" in line:
                    self.num_link = int(line[-1])
                if "~" and "capacity" in line:
                    lines = lines[i + 1:]
                    break
        # create NODE and LINK instance
        self.node_set = [Node(i) for i in range(self.num_node + 1)]
        self.link_set = [Link(0)]
        for i, line in enumerate(lines):
            tail, head = self.node_set[int(line[0])], self.node_set[int(line[1])]
            link = Link(i+1, tail, head, float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]))
            self.link_set.append(link)
            tail.link_out.append(link)
            head.link_in.append(link)

    def read_OD(self) -> None:
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_trips.txt") as f:
            lines = f.readlines()
            pattern = re.compile(r'[a-zA-Z0-9.]+')
            lines = [pattern.findall(line) for line in lines if pattern.findall(line) != []]
            for i, line in enumerate(lines):
                if "TOTAL" in line:
                    total_flow = float(line[-1])
                if "Origin" in line:
                    lines = lines[i:]
                    break
            for line in lines:
                if "Origin" in line:
                    origin = self.node_set[int(line[-1])]
                else:
                    for i in range(len(line) // 2):
                        destination = self.node_set[int(line[2 * i])]
                        if origin == destination:
                            total_flow -= float(line[2 * i + 1])
                            continue
                        demand = float(line[2 * i + 1])
                        if demand != .0:
                            self.od_set.append(OD(origin, destination, demand * self.demand_level, self))
            read_demand = int(sum([od.demand for od in self.od_set]))
            if not read_demand * 0.99 <= int(total_flow * self.demand_level) <= read_demand * 1.01:
                raise ValueError("Inconsistent demand data !")
            self.total_flow = total_flow
            self.num_od = len(self.od_set)
    
    def read_flow(self) -> None:
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_flow.txt") as file:
            lines = file.readlines()
            pattern = re.compile(r'[0-9.]+')
            data = list()
            for line in lines:
                line = pattern.findall(line)
                if len(line) != 0:
                    data.append([float(line[-2]), float(line[-1])])
            for index, value in enumerate(data):
                self.link_set[index+1].flow = value[0]
                self.link_set[index+1].cost = value[1]
        self.update_all_link_cost_and_marginal_cost()

    def update_all_link_cost(self) -> None:
        for link in self.link_set[1:]:
            link.update_cost()

    def update_all_link_marginal_cost(self) -> None:
        for link in self.link_set[1:]:
            link.update_marginal_cost()

    def update_all_link_cost_and_marginal_cost(self) -> None:
        self.update_all_link_cost()
        self.update_all_link_marginal_cost()

    def reset_network(self):
        for link in self.link_set[1:]:
            link.flow, link.aux_flow = 0, 0
            link.update_cost_and_marginal_cost()
        for od in self.od_set:
            od.working_set = list()
    
    @property
    def tstt(self):
        return sum([link.cost * link.flow for link in self.link_set[1:]])


if __name__ == '__main__':
    sf = Network("SiouxFalls")
    print(sf.od_set[22].shortest_path_and_cost())
