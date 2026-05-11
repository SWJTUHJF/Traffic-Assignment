#include "traffic_assignment/solver_base.hpp"

#include "traffic_assignment/shortest_path.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ta {

double relative_gap(Network& network, CostType cost_type) {
    double denominator = 0.0;
    for (const auto& od : network.od_set) {
        const auto sp = shortest_path(network, od.origin_id, od.destination_id, cost_type);
        denominator += od.demand * sp.cost;
    }

    if (denominator == 0.0) {
        return 0.0;
    }

    const double numerator = cost_type == CostType::Cost ? network.tstt() : network.tsmtt();
    return numerator / denominator - 1.0;
}

void BaseSolver::solve(Network& network, CostType cost_type, double tol_gap, bool verbose) {
    network_ = &network;
    cost_type_ = cost_type;
    tol_gap_ = tol_gap;
    verbose_ = verbose;
    cur_gap_ = std::numeric_limits<double>::infinity();
    iter_times_ = 0;

    preprocess();
    initialize();

    while (!check_terminate()) {
        ++iter_times_;
        main_loop_step();
        cur_gap_ = compute_gap();
        if (verbose_) {
            std::cout << "Iteration " << iter_times_
                      << ": current gap = " << std::scientific << std::setprecision(1) << cur_gap_
                      << ", TSTT = " << std::fixed << std::setprecision(1) << network_->tstt()
                      << '\n';
        }
    }

    postprocess();
    report();
}

double BaseSolver::current_gap() const {
    return cur_gap_;
}

int BaseSolver::iterations() const {
    return iter_times_;
}

void BaseSolver::preprocess() {
    start_time_ = std::chrono::steady_clock::now();
}

double BaseSolver::compute_gap() {
    if (network_ == nullptr) {
        throw std::runtime_error("Network is not set.");
    }
    return relative_gap(*network_, cost_type_);
}

bool BaseSolver::check_terminate() const {
    return cur_gap_ <= tol_gap_;
}

void BaseSolver::postprocess() {
    if (network_ == nullptr) {
        return;
    }
    if (cost_type_ == CostType::MarginalCost) {
        network_->update_all_link_cost();
    } else {
        network_->update_all_link_marginal_cost();
    }
}

void BaseSolver::report() const {
    if (network_ == nullptr) {
        return;
    }

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_);
    std::cout << "Running time = " << std::fixed << std::setprecision(5) << elapsed.count()
              << "s, TSTT = " << std::setprecision(1) << network_->tstt() << '\n';
}

}  // namespace ta
