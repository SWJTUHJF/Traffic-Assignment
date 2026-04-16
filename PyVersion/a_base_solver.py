from __future__ import annotations

from time import perf_counter as pc
from typing import Literal

from ..graph.network import Network

CostType = Literal["c", "mc"]


def relative_gap(network: Network, cost_type: CostType = "c") -> float:
    denominator = 0.0
    for od in network.od_set:
        sp = od.shortest_path(cost_type)
        denominator += sp.cost * od.demand

    if denominator == 0.0:
        return 0.0

    numerator = network.tstt if cost_type == "c" else sum(link.flow * link.marginal_cost for link in network.link_set)
    return numerator / denominator - 1.0


class BaseSolver:
    def __init__(self) -> None:
        self.network: Network | None = None
        self.cost_type: CostType = "c"
        self.tol_gap: float = 1e-4
        self.verbose: bool = False
        self.cur_gap: float = float("inf")
        self.iter_times: int = 0
        self.start_time: float = 0.0

    def solve(
        self,
        network: Network,
        cost_type: CostType = "c",
        tol_gap: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        self.network = network
        self.cost_type = cost_type
        self.tol_gap = tol_gap
        self.verbose = verbose
        self.cur_gap = float("inf")
        self.iter_times = 0

        self.preprocess()
        self.initialize()

        while not self.check_terminate():
            self.main_loop_step()
            self.iter_times += 1
            self.cur_gap = self.compute_gap()
            if self.verbose:
                print(f"Iteration {self.iter_times}: current gap = {self.cur_gap:.8f}")

        self.postprocess()
        self.report()

    def preprocess(self) -> None:
        self.start_time = pc()

    def initialize(self) -> None:
        raise NotImplementedError

    def main_loop_step(self) -> None:
        raise NotImplementedError

    def compute_gap(self) -> float:
        if self.network is None:
            return float("inf")
        return relative_gap(self.network, self.cost_type)

    def check_terminate(self) -> bool:
        return self.cur_gap <= self.tol_gap

    def postprocess(self) -> None:
        pass

    def report(self) -> None:
        if self.network is None:
            return
        print(f"Running time = {pc() - self.start_time:.5f}s, TSTT = {self.network.tstt:.5f}")
