#include "traffic_assignment/link_based.hpp"

#include "traffic_assignment/shortest_path.hpp"

#include <stdexcept>

namespace ta {

void FrankWolfe::run_FW_UE(Network& network, double tol_gap, bool verbose) {
    solve(network, CostType::Cost, tol_gap, verbose);
}

void FrankWolfe::run_FW_SO(Network& network, double tol_gap, bool verbose) {
    solve(network, CostType::MarginalCost, tol_gap, verbose);
}

void FrankWolfe::initialize() {
    if (network_ == nullptr) {
        throw std::runtime_error("Network is not set.");
    }

    all_or_nothing();
    for (auto& link : network_->link_set) {
        link.flow = link.aux_flow;
    }
}

void FrankWolfe::main_loop_step() {
    if (network_ == nullptr) {
        throw std::runtime_error("Network is not set.");
    }

    if (cost_type_ == CostType::Cost) {
        network_->update_all_link_cost();
    } else {
        network_->update_all_link_marginal_cost();
    }

    all_or_nothing();
    const double step = bisection();
    update_flow(step);
}

void FrankWolfe::all_or_nothing() {
    for (auto& link : network_->link_set) {
        link.aux_flow = 0.0;
    }

    for (const auto& od : network_->od_set) {
        const auto path = shortest_path(*network_, od.origin_id, od.destination_id, cost_type_);
        for (int link_id : path.link_ids) {
            network_->link_set.at(static_cast<std::size_t>(link_id)).aux_flow += od.demand;
        }
    }
}

void FrankWolfe::update_flow(double step) {
    for (auto& link : network_->link_set) {
        link.flow = link.flow + step * (link.aux_flow - link.flow);
    }
}

double FrankWolfe::bisection() const {
    double left = 0.0;
    double mid = 0.5;
    double right = 1.0;

    for (int i = 0; i < 30; ++i) {
        if (derivative(mid) * derivative(right) > 0.0) {
            right = mid;
        } else {
            left = mid;
        }
        mid = (left + right) / 2.0;
    }

    return mid;
}

double FrankWolfe::derivative(double alpha) const {
    double value = 0.0;
    for (const auto& link : network_->link_set) {
        const double direction = link.aux_flow - link.flow;
        const double trial_flow = link.flow + alpha * direction;
        if (cost_type_ == CostType::Cost) {
            value += link.get_cost(trial_flow) * direction;
        } else {
            value += link.get_marginal_cost(trial_flow) * direction;
        }
    }
    return value;
}

}  // namespace ta
