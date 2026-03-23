from load_network import Network, Link

import heapq
from collections import deque
from math import inf

import matplotlib.pyplot as plt



"""
Shortest path algorithm
    [GLC, LC, LS, LSF]
    For searching shortest marginal cost, use LS_SO
"""


__all__ = ["spp_algorithm_list", "obtain_list_link_costs", "obtain_list_link_marginal_cost", "plot_convergence"]


def initialize(network: Network, o_id: int) -> None:
    # Initialize parameters
    for node in network.node_set:
        node.parent = node
        node.dist = inf
        node.visited = False
    network.node_set[o_id].dist = 0
    network.node_set[o_id].parent = -1
    network.node_set[o_id].visited = True


# Based on the current parent label, obtain the shortest path.
def obtain_shortest_path(network: Network, d_id: int) -> list[Link]:
    shortest_path_links, current_node = list(), network.node_set[d_id]
    while current_node.parent != -1:
        for link in current_node.upstream_links:
            if link.tail == current_node.parent:
                shortest_path_links.append(link)
                break
        else:
            return []
        current_node = current_node.parent
    return shortest_path_links[::-1]


# GLC or Bellman-Ford algorithm
def GLC(
        network: Network,
        o_id: int,
        d_id: int,
        forbidden_nodes = None,
        forbidden_links = None
) -> list[Link]:
    initialize(network, o_id)
    # 循环num_node-1次就一定能够找到最短路
    for _ in network.node_set[2:]:
        updated = False
        # 对每条弧进行检查，head节点dist是否能够更新
        for link in network.link_set[1:]:
            tail_node, head_node = link.tail, link.head
            if forbidden_nodes and (head_node in forbidden_nodes):
                continue
            if forbidden_links and (link == forbidden_links):
                continue
            if tail_node.dist != inf and head_node.dist > tail_node.dist + link.cost:
                head_node.dist = tail_node.dist + link.cost
                head_node.parent = tail_node
                updated = True
        if not updated:
            break
    return obtain_shortest_path(network, d_id)


# LC or SPFA algorithm
def LC(
        network: Network,
        o_id: int,
        d_id: int,
        forbidden_nodes = None,
        forbidden_links = None
) -> list[Link]:
    initialize(network, o_id)
    SEL = deque([network.node_set[o_id]])
    while len(SEL):
        cur_node = SEL.popleft()
        cur_node.visited = False
        for link in cur_node.downstream_links:
            next_node = link.head
            if (forbidden_nodes and (next_node in forbidden_nodes)) or (forbidden_links and (link == forbidden_links)):
                continue
            if cur_node.dist != inf and next_node.dist > cur_node.dist + link.cost:
                next_node.dist = cur_node.dist + link.cost
                next_node.parent = cur_node
                if not next_node.visited:
                    SEL.append(next_node)
    return obtain_shortest_path(network, d_id)


# LS or Dijkstra algorithm
def LS(
        network: Network,
        o_id: int,
        d_id: int,
        forbidden_nodes = None,
        forbidden_links = None
) -> list[Link]:
    initialize(network, o_id)
    SEL = [network.node_set[o_id]]
    while SEL:
        SEL.sort(key=lambda node: node.dist, reverse=True)
        cur_node = SEL.pop()
        if cur_node == network.node_set[d_id]:
            break
        for link in cur_node.downstream_links:
            next_node, dist = link.head, link.cost
            if (forbidden_nodes and (next_node in forbidden_nodes)) or (forbidden_links and (link == forbidden_links)):
                continue
            if next_node.dist > cur_node.dist + dist:
                next_node.dist = cur_node.dist + dist
                next_node.parent = cur_node
                if next_node not in SEL:
                    SEL.append(next_node)
    return obtain_shortest_path(network, d_id)


# Dijkstra priority queue algorithm
def LSF(
        network: Network,
        o_id: int,
        d_id: int,
        forbidden_nodes = None,
        forbidden_links = None
) -> list[Link]:
    initialize(network, o_id)
    SEL = list()
    heapq.heappush(SEL, (network.node_set[o_id].dist, o_id))
    while SEL:
        cur_dist, cur_node = heapq.heappop(SEL)
        cur_node = network.node_set[cur_node]
        if cur_node == network.node_set[d_id]:
            break
        for link in cur_node.downstream_links:
            next_node, dist = link.head, link.cost
            if (forbidden_nodes and (next_node in forbidden_nodes)) or (forbidden_links and (link == forbidden_links)):
                continue
            if next_node.dist > cur_node.dist + dist:
                next_node.dist = cur_node.dist + dist
                next_node.parent = cur_node
                if (next_node.dist, next_node.node_id) not in SEL:
                    heapq.heappush(SEL, (next_node.dist, next_node.node_id))
    return obtain_shortest_path(network, d_id)


def obtain_list_link_costs(list_link: list[Link]) -> float:
    return sum([link.cost for link in list_link])


def obtain_list_link_marginal_cost(list_link: list[Link]) -> float:
    return sum([link.marginal_cost for link in list_link])


# For SO
def LS_SO(
        network: Network,
        o_id: int,
        d_id: int
) -> list[Link]:
    initialize(network, o_id)
    # main loop
    SEL = [network.node_set[o_id]]
    while SEL:
        SEL.sort(key=lambda n: n.dist, reverse=True)
        cur = SEL.pop()
        if cur.node_id == d_id:
            break
        for link in cur.downstream_links:
            if link.head.dist > cur.dist + link.marginal_cost:
                link.head.parent = cur
                link.head.dist = cur.dist + link.marginal_cost
                if link.head not in SEL:
                    SEL.append(link.head)
    return obtain_shortest_path(network, d_id)


def k_shortest_path(
        network: Network,
        o_id: int,
        d_id: int,
        k: int
) -> (list[list[Link]], int):
    k_paths = []  # to store the k-paths
    shortest_path, shortest_cost = LSF(network, o_id, d_id)
    k_paths.append(shortest_path)
    possible_paths = []  # to store the obtained possible paths
    existing_paths = [shortest_path]
    while len(k_paths) != k:
        previous_path = k_paths[-1]
        for spur_link_index, spur_link in enumerate(previous_path):
            spur_node = spur_link.tail.node_id  # deviate for each link's tail node
            root_path = previous_path[:spur_link_index]
            # calculate the cost of the root path
            root_cost = sum([link.cost for link in root_path])
            # remove the adjacent link and root nodes
            forbidden_nodes = [link.tail for link in root_path]
            forbidden_links = spur_link
            # find the spur path
            spur_path = LSF(network, spur_node, d_id, forbidden_nodes, forbidden_links)
            spur_cost = obtain_list_link_costs(spur_path)
            total_path = root_path + spur_path
            if (spur_path is None) or (total_path in existing_paths):
                continue
            else:
                total_cost = root_cost + spur_cost
                existing_paths.append(total_path)
                heapq.heappush(possible_paths, (total_cost, total_path))
        if not possible_paths:
            return k_paths, len(k_paths)
        cost, path = heapq.heappop(possible_paths)
        k_paths.append(path)
    return k_paths, len(k_paths)


def plot_convergence(gap_list_dict: dict[(float, float): list[float]]):
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, val in gap_list_dict.items():
        ax.plot(val, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Current Gap", fontsize=14)
    ax.grid()
    ax.legend()
    plt.show()


spp_algorithm_list = {"GLC": GLC,
                      "LC": LC,
                      "LS": LS,
                      "LSF": LSF,
                      "LS_SO": LS_SO,
                      "k_shortest_path": k_shortest_path}


if __name__ == '__main__':
    sf = Network("SiouxFalls")
    print(LS(sf, 1, 24))
