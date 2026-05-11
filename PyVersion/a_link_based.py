from a_base_solver import BaseSolver
from g_network import Network


class MSA(BaseSolver):
    def run_MSA_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, kind= "tt", tol_gap=tol_gap, verbose=verbose)

    def run_MSA_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, kind="mtt", tol_gap=tol_gap, verbose=verbose)
    
    def preprocess(self) -> None:
        super().preprocess()
        self.network.reset_assignment()
    
    def initialize(self) -> None:
        self.all_or_nothing(self.network)
        for link in self.network.link_set:
            link.flow = link.aux_flow

    def main_loop_step(self) -> None:
        self.network.update_all_link_cost(self.kind)
        self.all_or_nothing(self.network)
        step = 1 / (self.iter_times + 1)
        self.update_flow(step, self.network)

    def update_flow(self, step: float, network: Network) -> None:
        for link in network.link_set:
            link.flow = link.flow + step * (link.aux_flow - link.flow)

    def all_or_nothing(self, network: Network) -> None:
        for link in network.link_set:
            link.aux_flow = 0.0

        for od in network.od_set:
            path = od.shortest_path(kind=self.kind)
            for link in path:
                link.aux_flow += od.demand


class FrankWolfe(BaseSolver):
    def run_FW_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, kind="tt", tol_gap=tol_gap, verbose=verbose)

    def run_FW_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, kind="mtt", tol_gap=tol_gap, verbose=verbose)

    def initialize(self) -> None:
        self.all_or_nothing(self.network)
        for link in self.network.link_set:
            link.flow = link.aux_flow

    def main_loop_step(self) -> None:
        self.network.update_all_link_cost(self.kind)
        self.all_or_nothing(self.network)
        step = self.bisection(self.network)
        self.update_flow(step, self.network)

    def update_flow(self, step: float, network: Network) -> None:
        for link in network.link_set:
            link.flow = link.flow + step * (link.aux_flow - link.flow)

    def all_or_nothing(self, network: Network) -> None:
        for link in network.link_set:
            link.aux_flow = 0.0

        for od in network.od_set:
            path = od.shortest_path(kind=self.kind)
            for link in path:
                link.aux_flow += od.demand

    def bisection(self, network: Network) -> float:
        def derivative(alpha: float) -> float:
            return sum(
                    link.get_specific_cost(link.flow + alpha * (link.aux_flow - link.flow), self.kind) * (link.aux_flow - link.flow)
                    for link in network.link_set
                )

        left, mid, right = 0.0, 0.5, 1.0
        for _ in range(30):
            if derivative(mid) * derivative(right) > 0.0:
                right = mid
            else:
                left = mid
            mid = (left + right) / 2.0
        return mid

class ConjugateFrankWolfe(BaseSolver):
    pass


class BiConjugateFrankWolfe(BaseSolver):
    pass
