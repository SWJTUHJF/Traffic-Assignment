from __future__ import annotations

from time import perf_counter as pc
from typing import Literal

from g_network import Network

CostType = Literal["c", "mc"]


def relative_gap(network: Network, cost_type: CostType = "c") -> float:
    denominator = 0.0
    for od in network.od_set:
        sp = od.shortest_path(cost_type)
        add_value = sp.cost if cost_type == "c" else sp.marginal_cost
        denominator += od.demand * add_value

    if denominator == 0.0:
        return 0.0

    numerator = network.tstt if cost_type == "c" else network.tsmtt
    return numerator / denominator - 1.0


class BaseSolver:
    def __init__(self):
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
            self.iter_times += 1
            self.main_loop_step()
            self.cur_gap = self.compute_gap()
            if self.verbose:
                print(f"Iteration {self.iter_times}: current gap = {self.cur_gap:.1e}, TSTT = {self.network.tstt:.1f}")

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
            raise ValueError("Network is not set.")
        return relative_gap(self.network, self.cost_type)

    def check_terminate(self) -> bool:
        return self.cur_gap <= self.tol_gap

    def postprocess(self) -> None:
        if self.cost_type == "mc":
            self.network.update_all_link_cost()
        if self.cost_type == "c":
            self.network.update_all_link_marginal_cost()

    def report(self) -> None:
        if self.network is None:
            return
        print(f"Running time = {pc() - self.start_time:.5f}s, TSTT = {self.network.tstt:.1f}")
