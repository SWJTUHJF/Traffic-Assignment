from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from g_network import Link, Network, Node


@dataclass
class SearchResult:
    origin: Node
    dist: dict
    prev_link: dict
    _resticted: bool

    def path_to(self, destination) -> list[Link]:
        if self.prev_link[destination] is None:
            if not self._resticted:
                return []
            raise ValueError(f"No path from {self.origin.node_id} to {destination.node_id}")

        path, node = [], destination
        while node is not self.origin:
            link = self.prev_link[node]
            path.append(link)
            node = link.tail

        path.reverse()
        return path


def dijkstra(
    network: Network,
    origin: Node,
    destination: Node | None = None,
    cost_type: Literal["c", "mc"] = "c",
    resticted: bool = True,
    pre_terminate: bool = True,
) -> SearchResult:
    dist: dict[Node, float] = {node: float('inf') for node in network.node_set}
    prev_link: dict[Node, Link | None] = {node: None for node in network.node_set}

    dist[origin] = 0.0
    pq: list[tuple[float, int, Node]] = [(0.0, origin.node_id, origin)]

    while pq:
        current_dist, _, current = heappop(pq)
        if current_dist > dist[current]:
            continue
        if destination is not None and current is destination and pre_terminate:
            break

        for link in current.link_out:
            edge_cost = link.cost if cost_type == "c" else link.marginal_cost
            nxt = link.head
            proposal = current_dist + edge_cost
            if proposal < dist[nxt]:
                dist[nxt] = proposal
                prev_link[nxt] = link
                heappush(pq, (proposal, nxt.node_id, nxt))

    return SearchResult(origin, dist, prev_link, resticted)
