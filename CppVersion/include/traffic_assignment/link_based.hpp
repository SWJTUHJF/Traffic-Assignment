#pragma once

#include "traffic_assignment/solver_base.hpp"

namespace ta {

class FrankWolfe : public BaseSolver {
public:
    void run_FW_UE(Network& network, double tol_gap = 1e-4, bool verbose = false);
    void run_FW_SO(Network& network, double tol_gap = 1e-4, bool verbose = false);

protected:
    void initialize() override;
    void main_loop_step() override;

private:
    void all_or_nothing();
    void update_flow(double step);
    [[nodiscard]] double bisection() const;
    [[nodiscard]] double derivative(double alpha) const;
};

}  // namespace ta
