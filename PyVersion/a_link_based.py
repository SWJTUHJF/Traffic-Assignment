from __future__ import annotations

from .alg_base_solver import BaseSolver, CostType, relative_gap
from ..graph.network import Network


class LinkBased(BaseSolver):
    def run_FW_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_FW_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)

    def run_CFW_UE(self) -> None:
        raise NotImplementedError

    def run_CFW_SO(self) -> None:
        raise NotImplementedError

    def run_BCFW_UE(self) -> None:
        raise NotImplementedError

    def run_BCFW_SO(self) -> None:
        raise NotImplementedError

    def initialize(self) -> None:
        assert self.network is not None
        self.all_or_nothing(self.network, self.cost_type)
        for link in self.network.link_set:
            link.flow = link.aux_flow

    def main_loop_step(self) -> None:
        assert self.network is not None

        if self.cost_type == "c":
            self.network.update_all_link_cost()
        else:
            self.network.update_all_link_marginal_cost()

        self.all_or_nothing(self.network, self.cost_type)
        step = self.bisection(self.network, self.cost_type)
        self.update_flow(step, self.network)

    def postprocess(self) -> None:
        assert self.network is not None
        if self.cost_type == "mc":
            self.network.update_all_link_cost()

    def update_flow(self, step: float, network: Network) -> None:
        for link in network.link_set:
            link.flow = link.flow + step * (link.aux_flow - link.flow)

    def all_or_nothing(self, network: Network, cost_type: CostType) -> None:
        for link in network.link_set:
            link.aux_flow = 0.0

        for od in network.od_set:
            path = od.shortest_path(cost_type=cost_type)
            for link in path:
                link.aux_flow += od.demand

    def bisection(self, network: Network, cost_type: CostType) -> float:
        def derivative(alpha: float) -> float:
            if cost_type == "c":
                return sum(
                    link.get_cost(link.flow + alpha * (link.aux_flow - link.flow)) * (link.aux_flow - link.flow)
                    for link in network.link_set
                )
            return sum(
                link.get_marginal_cost(link.flow + alpha * (link.aux_flow - link.flow))
                * (link.aux_flow - link.flow)
                for link in network.link_set
            )

        left = 0.0
        right = 1.0
        mid = 0.5
        for _ in range(30):
            if derivative(mid) * derivative(right) > 0.0:
                right = mid
            else:
                left = mid
            mid = (left + right) / 2.0
        return mid


__all__ = ["LinkBased", "CostType", "relative_gap"]
