from __future__ import annotations

from heapq import heappop, heappush
from math import inf
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from g_network import Link, Network, Node


def dijkstra(
    network: Network,
    origin: Node,
    destination: Node,
    cost_type: Literal["c", "mc"] = "c",
    resticted: bool = True,
) -> list[Link]:
    dist = {node: inf for node in network.node_set}
    prev_link: dict[Node, Link | None] = {node: None for node in network.node_set}

    dist[origin] = 0.0
    pq: list[tuple[float, int, Node]] = [(0.0, origin.node_id, origin)]

    while pq:
        current_dist, _, current = heappop(pq)
        if current_dist > dist[current]:
            continue
        if current is destination:
            break

        for link in current.link_out:
            edge_cost = link.cost if cost_type == "c" else link.marginal_cost
            nxt = link.head
            proposal = current_dist + edge_cost
            if proposal < dist[nxt]:
                dist[nxt] = proposal
                prev_link[nxt] = link
                heappush(pq, (proposal, nxt.node_id, nxt))

    if origin is not destination and prev_link[destination] is None:
        if not resticted:
            return []
        raise ValueError(f"No path from {origin.node_id} to {destination.node_id} in network {network.name}")

    path: list[Link] = []
    node = destination
    while node is not origin:
        link = prev_link[node]
        path.append(link)
        node = link.tail

    path.reverse()
    return path
