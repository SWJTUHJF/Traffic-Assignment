from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import TYPE_CHECKING

from utilities import CostType

if TYPE_CHECKING:
    from g_network import Link, Network, Node


@dataclass
class SearchResult:
    origin: Node
    dist: dict[Node, float]
    prev_link: dict[Node, Link | None]
    _resticted: bool = True  # whether allow the occurence of disconnectivity

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
    kind: CostType = "tt",
    resticted: bool = True,  # if restricted, raise error when no path exists
    pre_terminate: bool = True,  # true if only need one-to-to label
    forbidden_links: list[Link] | None = None,
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
            if forbidden_links is not None and link in forbidden_links:
                continue
            edge_cost = link.cost(kind=kind)
            nxt = link.head
            proposal = current_dist + edge_cost
            if proposal < dist[nxt]:
                dist[nxt] = proposal
                prev_link[nxt] = link
                heappush(pq, (proposal, nxt.node_id, nxt))

    return SearchResult(origin, dist, prev_link, resticted)


def nodes_from_links(links: list[Link]) -> list[Node]:
    nodes = [links[0].tail]
    for link in links:
        nodes.append(link.head)
    return nodes
