from __future__ import annotations
from math import inf
from typing import Literal
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from build_network import Network, Link, Node


def dijkstra(
        network: Network,
        origin: Node,
        destination: Node,
        cost_type: Literal["c", "mc"] = "c",
        ) -> list[Link]:
    # initialize
    dist = {node: inf for node in network.node_set[1:]}
    parent = {node: None for node in network.node_set[1:]}
    dist[origin], parent[origin] = 0, -1
    # main loop
    SEL = [origin]
    while SEL:
        SEL.sort(key=lambda n: dist[n], reverse=True)
        current_node = SEL.pop()
        if current_node == destination:
            break
        for link in current_node.link_out:
            link_cost = link.cost if cost_type == "c" else link.marginal_cost
            head_node = link.head
            if dist[head_node] > dist[current_node] + link_cost:
                dist[head_node] = dist[current_node] + link_cost
                parent[head_node] = current_node
                if head_node not in SEL:
                    SEL.append(head_node)
    # obtain the shortest path
    shortest_path, current_node = [], destination
    while parent[current_node] != -1:
        p = parent[current_node]
        for link in current_node.link_in:
            if link.tail == p:
                shortest_path.append(link)
                current_node = p
                break
        else:
            raise ValueError("Parent node errors!")
    shortest_path.reverse()
    return shortest_path

