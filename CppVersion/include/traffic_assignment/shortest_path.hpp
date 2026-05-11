#pragma once

#include "traffic_assignment/network.hpp"

#include <vector>

namespace ta {

struct SearchResult {
    int origin_id{};
    std::vector<double> dist;
    std::vector<int> prev_link;
    bool restricted{true};

    [[nodiscard]] Path path_to(const Network& network, int destination_id) const;
};

[[nodiscard]] SearchResult dijkstra(
    const Network& network,
    int origin_id,
    int destination_id = -1,
    CostType cost_type = CostType::Cost,
    bool restricted = true,
    bool pre_terminate = true
);

[[nodiscard]] Path shortest_path(
    const Network& network,
    int origin_id,
    int destination_id,
    CostType cost_type = CostType::Cost
);

}  // namespace ta
