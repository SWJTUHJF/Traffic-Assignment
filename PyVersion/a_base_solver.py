from __future__ import annotations

from time import perf_counter as pc

from g_network import Network
from utilities import CostType



def relative_gap(network: Network, kind: CostType = "tt") -> float:
    denominator = 0.0
    for od in network.od_set:
        sp = od.shortest_path(kind)
        add_value = sp.get_cost(kind)
        denominator += od.demand * add_value

    if denominator == 0.0:
        return 0.0

    numerator = network.tstc(kind)
    return numerator / denominator - 1.0


class BaseSolver:
    def __init__(self):
        self.kind: CostType = "tt"
        self.tol_gap: float = 1e-4
        self.verbose: bool = False
        self.cur_gap: float = float("inf")
        self.iter_times: int = 0
        self.start_time: float = 0.0

    def solve(
        self,
        network: Network,
        kind: CostType = "tt",
        tol_gap: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        self.network = network
        self.kind = kind
        print(self.kind)
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
                print(f"Iteration {self.iter_times}: current gap = {self.cur_gap:.1e}, TSTC = {self.network.tstc(self.kind):.1f}")

        self.postprocess()
        self.report()

    def preprocess(self) -> None:
        self.start_time = pc()

    def initialize(self) -> None:
        raise NotImplementedError

    def main_loop_step(self) -> None:
        raise NotImplementedError

    def compute_gap(self) -> float:
        return relative_gap(self.network, self.kind)

    def check_terminate(self) -> bool:
        return self.cur_gap <= self.tol_gap

    def postprocess(self) -> None:
        self.network.update_all_link_cost("tt")

    def report(self) -> None:
        if self.network is None:
            return
        print(f"Running time = {pc() - self.start_time:.5f}s, TSTT = {self.network.tstc("tt"):.1f}")
