#include "traffic_assignment/shortest_path.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace ta {

Path SearchResult::path_to(const Network& network, int destination_id) const {
    if (destination_id < 0 || destination_id >= static_cast<int>(prev_link.size())) {
        throw std::out_of_range("Destination id is outside node range.");
    }

    if (prev_link.at(static_cast<std::size_t>(destination_id)) < 0) {
        if (!restricted) {
            return Path{origin_id, destination_id, {}, dist.at(static_cast<std::size_t>(destination_id))};
        }
        throw std::runtime_error("No path from origin to destination.");
    }

    std::vector<int> path_links;
    int node_id = destination_id;
    while (node_id != origin_id) {
        const int link_id = prev_link.at(static_cast<std::size_t>(node_id));
        if (link_id < 0) {
            throw std::runtime_error("Broken predecessor chain in shortest path result.");
        }
        path_links.push_back(link_id);
        node_id = network.link_set.at(static_cast<std::size_t>(link_id)).tail_id;
    }

    std::reverse(path_links.begin(), path_links.end());
    return Path{
        origin_id,
        destination_id,
        std::move(path_links),
        dist.at(static_cast<std::size_t>(destination_id))
    };
}

SearchResult dijkstra(
    const Network& network,
    int origin_id,
    int destination_id,
    CostType cost_type,
    bool restricted,
    bool pre_terminate
) {
    if (origin_id < 0 || origin_id >= network.num_node) {
        throw std::out_of_range("Origin id is outside node range.");
    }
    if (destination_id >= network.num_node) {
        throw std::out_of_range("Destination id is outside node range.");
    }

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> dist(static_cast<std::size_t>(network.num_node), inf);
    std::vector<int> prev_link(static_cast<std::size_t>(network.num_node), -1);

    using QueueItem = std::pair<double, int>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<>> pq;

    dist.at(static_cast<std::size_t>(origin_id)) = 0.0;
    pq.emplace(0.0, origin_id);

    while (!pq.empty()) {
        const auto [current_dist, current_id] = pq.top();
        pq.pop();

        if (current_dist > dist.at(static_cast<std::size_t>(current_id))) {
            continue;
        }
        if (destination_id >= 0 && current_id == destination_id && pre_terminate) {
            break;
        }

        const auto& current = network.node_set.at(static_cast<std::size_t>(current_id));
        for (int link_id : current.link_out) {
            const auto& link = network.link_set.at(static_cast<std::size_t>(link_id));
            const int next_id = link.head_id;
            const double proposal = current_dist + link.weight(cost_type);
            if (proposal < dist.at(static_cast<std::size_t>(next_id))) {
                dist.at(static_cast<std::size_t>(next_id)) = proposal;
                prev_link.at(static_cast<std::size_t>(next_id)) = link_id;
                pq.emplace(proposal, next_id);
            }
        }
    }

    return SearchResult{origin_id, std::move(dist), std::move(prev_link), restricted};
}

Path shortest_path(const Network& network, int origin_id, int destination_id, CostType cost_type) {
    return dijkstra(network, origin_id, destination_id, cost_type, true, true).path_to(network, destination_id);
}

}  // namespace ta
