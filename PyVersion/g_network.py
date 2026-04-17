from typing import Literal

from g_sp import dijkstra


CostType = Literal["c", "mc"]


class Node:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.link_in: list[Link] = []
        self.link_out: list[Link] = []

    def __repr__(self) -> str:
        return f"Node {self.node_id+1}"


class Link:
    def __init__(
        self,
        link_id: int,
        tail: Node,
        head: Node,
        capacity: float,
        length: float,
        free_flow_time: float,
        alpha: float,
        beta: float,
    ):
        self.link_id = link_id
        self.tail = tail
        self.head = head
        self.capacity = capacity
        self.length = length
        self.fft = free_flow_time
        self.alpha = alpha
        self.beta = beta

        self.flow = 0.0
        self.aux_flow = 0.0
        self.cost: float = self.fft
        self.marginal_cost: float = self.fft

    def __repr__(self) -> str:
        return f"Link {self.link_id+1}({self.tail}->{self.head}): flow={self.flow:.1f}"

    def get_cost(self, flow: float) -> float:
        return self.fft * (1.0 + self.alpha * (flow / self.capacity) ** self.beta)

    def get_marginal_cost(self, flow: float) -> float:
        return self.get_cost(flow) + self.fft * self.alpha * self.beta * (flow / self.capacity) ** self.beta
    
    def update_cost(self) -> None:
        self.cost = self.get_cost(self.flow)

    def update_marginal_cost(self) -> None:
        self.marginal_cost = self.get_marginal_cost(self.flow)

    def update_cost_and_marginal_cost(self) -> None:
        self.update_cost()
        self.update_marginal_cost()

    @property
    def d_cost(self) -> float:
        return self.fft * self.alpha * self.beta * self.flow ** (self.beta - 1) / self.capacity ** self.beta

    @property
    def d_marginal_cost(self) -> float:
        return (self.alpha * self.fft * self.flow ** (self.beta - 1) * self.beta * (1 + self.beta) / self.capacity ** self.beta)


class Path:
    def __init__(self, origin: Node, destination: Node, included_links: list[Link]):
        self.origin = origin
        self.destination = destination
        self.included_links = included_links
        self.flow: float = 0.0

    def __iter__(self):
        return iter(self.included_links)

    def __hash__(self) -> int:
        return hash(tuple(link.link_id for link in self.included_links))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Path):
            return False
        return self.included_links == other.included_links

    def __repr__(self) -> str:
        nodes = [link.tail.node_id+1 for link in self.included_links] + [self.destination.node_id+1]
        return f"{'-'.join(map(str, nodes))}(cost={self.cost:.1f}, flow={self.flow:.1f})"

    def add_flow(self, value: float) -> None:
        self.flow += value
        for link in self.included_links:
            link.flow += value

    @property
    def cost(self) -> float:
        return sum(link.cost for link in self.included_links)

    @property
    def marginal_cost(self) -> float:
        return sum(link.marginal_cost for link in self.included_links)


class Network:
    def __init__(self, name: str, demand_level: float = 1.0):
        self.name = name
        self.demand_level = demand_level

        self.node_set: list[Node] = []
        self.link_set: list[Link] = []
        self.od_set: list[OD] = []

        self.num_node: int = 0
        self.num_link: int = 0
        self.num_od: int = 0
        self.total_flow: float = 0.0

    def add_link(self,
        tail_id: int,
        head_id: int,
        capacity: float,
        length: float,
        free_flow_time: float,
        alpha: float,
        beta: float,
    ) -> Link:
        tail = self.node_set[tail_id]
        head = self.node_set[head_id]
        link = Link(
            link_id=len(self.link_set),
            tail=tail,
            head=head,
            capacity=capacity,
            length=length,
            free_flow_time=free_flow_time,
            alpha=alpha,
            beta=beta,
        )
        self.link_set.append(link)
        tail.link_out.append(link)
        head.link_in.append(link)
        self.num_link = len(self.link_set)
        return link

    def add_od(self, origin_id: int, destination_id: int, demand: float) -> None:
        if demand <= 0.0 or origin_id == destination_id:
            return
        self.od_set.append(
            OD(
                origin=self.node_set[origin_id],
                destination=self.node_set[destination_id],
                demand=demand,
                network=self,
            )
        )
        self.num_od = len(self.od_set)

    def update_all_link_cost(self) -> None:
        for link in self.link_set:
            link.update_cost()

    def update_all_link_marginal_cost(self) -> None:
        for link in self.link_set:
            link.update_marginal_cost()

    def update_all_link_cost_and_marginal_cost(self) -> None:
        self.update_all_link_cost()
        self.update_all_link_marginal_cost()

    def reset_assignment(self) -> None:
        for link in self.link_set:
            link.flow = 0.0
            link.aux_flow = 0.0
            link.update_cost_and_marginal_cost()
        for od in self.od_set:
            od.working_set = []

    @property
    def tstt(self) -> float:
        return sum(link.cost * link.flow for link in self.link_set)
    
    @property
    def tsmtt(self) -> float:
        return sum(link.marginal_cost * link.flow for link in self.link_set)

    def shortest_path(self, origin: Node | int, destination: Node | int, cost_type: CostType = "c") -> Path:
        if isinstance(origin, int):
            origin = self.node_set[origin-1]
        if isinstance(destination, int):
            destination = self.node_set[destination-1]
        return Path(origin, destination, dijkstra(self, origin, destination, cost_type))

class OD:
    def __init__(self, origin: Node, destination: Node, demand: float, network: Network):
        self.origin = origin
        self.destination = destination
        self.demand = demand
        self.network = network
        self.working_set: list[Path] = []

    def __repr__(self) -> str:
        return f"OD {self.origin.node_id}->{self.destination.node_id}"

    def shortest_path(self, cost_type: CostType = "c") -> Path:
        return self.network.shortest_path(self.origin, self.destination, cost_type)
