#include "traffic_assignment/network.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ta {

Link::Link(
    int link_id,
    int tail_id,
    int head_id,
    double capacity,
    double length,
    double free_flow_time,
    double alpha,
    double beta
)
    : link_id(link_id),
      tail_id(tail_id),
      head_id(head_id),
      capacity(capacity),
      length(length),
      free_flow_time(free_flow_time),
      alpha(alpha),
      beta(beta),
      cost(free_flow_time),
      marginal_cost(free_flow_time) {}

double Link::get_cost(double x) const {
    if (capacity <= 0.0) {
        return free_flow_time;
    }
    return free_flow_time * (1.0 + alpha * std::pow(x / capacity, beta));
}

double Link::get_marginal_cost(double x) const {
    if (capacity <= 0.0) {
        return free_flow_time;
    }
    return get_cost(x) + free_flow_time * alpha * beta * std::pow(x / capacity, beta);
}

double Link::weight(CostType cost_type) const {
    return cost_type == CostType::Cost ? cost : marginal_cost;
}

void Link::update_cost() {
    cost = get_cost(flow);
}

void Link::update_marginal_cost() {
    marginal_cost = get_marginal_cost(flow);
}

void Link::update_cost_and_marginal_cost() {
    update_cost();
    update_marginal_cost();
}

std::vector<int> Path::node_ids(const std::vector<Link>& links) const {
    std::vector<int> nodes;
    if (link_ids.empty()) {
        return nodes;
    }

    nodes.push_back(links.at(static_cast<std::size_t>(link_ids.front())).tail_id);
    for (int link_id : link_ids) {
        nodes.push_back(links.at(static_cast<std::size_t>(link_id)).head_id);
    }
    return nodes;
}

Network::Network(std::string name, double demand_level)
    : name(std::move(name)), demand_level(demand_level) {}

void Network::resize_nodes(int count) {
    if (count < 0) {
        throw std::invalid_argument("Number of nodes cannot be negative.");
    }

    node_set.clear();
    node_set.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        node_set.emplace_back(i);
    }
    num_node = count;
}

Link& Network::add_link(
    int tail_id,
    int head_id,
    double capacity,
    double length,
    double free_flow_time,
    double alpha,
    double beta
) {
    if (tail_id < 0 || tail_id >= num_node || head_id < 0 || head_id >= num_node) {
        throw std::out_of_range("Link endpoint id is outside node range.");
    }

    const int link_id = static_cast<int>(link_set.size());
    link_set.emplace_back(link_id, tail_id, head_id, capacity, length, free_flow_time, alpha, beta);
    node_set.at(static_cast<std::size_t>(tail_id)).link_out.push_back(link_id);
    node_set.at(static_cast<std::size_t>(head_id)).link_in.push_back(link_id);
    num_link = static_cast<int>(link_set.size());
    return link_set.back();
}

void Network::add_od(int origin_id, int destination_id, double demand) {
    if (demand <= 0.0 || origin_id == destination_id) {
        return;
    }
    if (origin_id < 0 || origin_id >= num_node || destination_id < 0 || destination_id >= num_node) {
        throw std::out_of_range("OD endpoint id is outside node range.");
    }

    od_set.push_back(OD{origin_id, destination_id, demand});
    num_od = static_cast<int>(od_set.size());
}

void Network::update_all_link_cost() {
    for (auto& link : link_set) {
        link.update_cost();
    }
}

void Network::update_all_link_marginal_cost() {
    for (auto& link : link_set) {
        link.update_marginal_cost();
    }
}

void Network::update_all_link_cost_and_marginal_cost() {
    for (auto& link : link_set) {
        link.update_cost_and_marginal_cost();
    }
}

double Network::tstt() const {
    double value = 0.0;
    for (const auto& link : link_set) {
        value += link.cost * link.flow;
    }
    return value;
}

double Network::tsmtt() const {
    double value = 0.0;
    for (const auto& link : link_set) {
        value += link.marginal_cost * link.flow;
    }
    return value;
}

}  // namespace ta
