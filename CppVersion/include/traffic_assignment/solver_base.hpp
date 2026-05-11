#pragma once

#include "traffic_assignment/network.hpp"

#include <chrono>

namespace ta {

[[nodiscard]] double relative_gap(Network& network, CostType cost_type = CostType::Cost);

class BaseSolver {
public:
    virtual ~BaseSolver() = default;

    void solve(
        Network& network,
        CostType cost_type = CostType::Cost,
        double tol_gap = 1e-4,
        bool verbose = false
    );

    [[nodiscard]] double current_gap() const;
    [[nodiscard]] int iterations() const;

protected:
    Network* network_{nullptr};
    CostType cost_type_{CostType::Cost};
    double tol_gap_{1e-4};
    bool verbose_{false};
    double cur_gap_{};
    int iter_times_{};
    std::chrono::steady_clock::time_point start_time_;

    virtual void initialize() = 0;
    virtual void main_loop_step() = 0;

    virtual void preprocess();
    virtual double compute_gap();
    virtual bool check_terminate() const;
    virtual void postprocess();
    virtual void report() const;
};

}  // namespace ta
