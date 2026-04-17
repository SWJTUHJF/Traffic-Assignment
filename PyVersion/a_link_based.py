from a_base_solver import BaseSolver
from g_network import Network


class MSA(BaseSolver):
    def run_MSA_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_MSA_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)

    def initialize(self) -> None:
        self.all_or_nothing(self.network)
        for link in self.network.link_set:
            link.flow = link.aux_flow

    def main_loop_step(self) -> None:
        if self.cost_type == "c":
            self.network.update_all_link_cost()
        else:
            self.network.update_all_link_marginal_cost()

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
            path = od.shortest_path(cost_type=self.cost_type)
            for link in path:
                link.aux_flow += od.demand


class FrankWolfe(BaseSolver):
    def run_FW_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_FW_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)

    def initialize(self) -> None:
        self.all_or_nothing(self.network)
        for link in self.network.link_set:
            link.flow = link.aux_flow

    def main_loop_step(self) -> None:
        if self.cost_type == "c":
            self.network.update_all_link_cost()
        else:
            self.network.update_all_link_marginal_cost()

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
            path = od.shortest_path(cost_type=self.cost_type)
            for link in path:
                link.aux_flow += od.demand

    def bisection(self, network: Network) -> float:
        def derivative(alpha: float) -> float:
            if self.cost_type == "c":
                return sum(
                    link.get_cost(link.flow + alpha * (link.aux_flow - link.flow)) * (link.aux_flow - link.flow)
                    for link in network.link_set
                )
            return sum(
                link.get_marginal_cost(link.flow + alpha * (link.aux_flow - link.flow))
                * (link.aux_flow - link.flow)
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
    def run_CFW_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_CFW_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)

    def initialize(self) -> None:
        self.all_or_nothing(self.network)
        for link in self.network.link_set:
            link.flow = link.aux_flow

    def main_loop_step(self) -> None:
        if self.cost_type == "c":
            self.network.update_all_link_cost()
        else:
            self.network.update_all_link_marginal_cost()

        self.all_or_nothing(self.network)
        step = self.bisection(self.network)
        self.update_flow(step, self.network)

    def postprocess(self) -> None:
        if self.cost_type == "mc":
            self.network.update_all_link_cost()
        if self.cost_type == "c":
            self.network.update_all_link_marginal_cost()

    def update_flow(self, step: float, network: Network) -> None:
        for link in network.link_set:
            link.flow = link.flow + step * (link.aux_flow - link.flow)

    def all_or_nothing(self, network: Network) -> None:
        for link in network.link_set:
            link.aux_flow = 0.0

        for od in network.od_set:
            path = od.shortest_path(cost_type=self.cost_type)
            for link in path:
                link.aux_flow += od.demand

    def bisection(self, network: Network) -> float:
        def derivative(alpha: float) -> float:
            if self.cost_type == "c":
                return sum(
                    link.get_cost(link.flow + alpha * (link.aux_flow - link.flow)) * (link.aux_flow - link.flow)
                    for link in network.link_set
                )
            return sum(
                link.get_marginal_cost(link.flow + alpha * (link.aux_flow - link.flow))
                * (link.aux_flow - link.flow)
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


class BiConjugateFrankWolfe(BaseSolver):
    def run_BCFW_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_BCFW_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)

    def initialize(self) -> None:
        self.all_or_nothing(self.network)
        for link in self.network.link_set:
            link.flow = link.aux_flow

    def main_loop_step(self) -> None:
        if self.cost_type == "c":
            self.network.update_all_link_cost()
        else:
            self.network.update_all_link_marginal_cost()

        self.all_or_nothing(self.network)
        step = self.bisection(self.network)
        self.update_flow(step, self.network)

    def postprocess(self) -> None:
        if self.cost_type == "mc":
            self.network.update_all_link_cost()
        if self.cost_type == "c":
            self.network.update_all_link_marginal_cost()

    def update_flow(self, step: float, network: Network) -> None:
        for link in network.link_set:
            link.flow = link.flow + step * (link.aux_flow - link.flow)

    def all_or_nothing(self, network: Network) -> None:
        for link in network.link_set:
            link.aux_flow = 0.0

        for od in network.od_set:
            path = od.shortest_path(cost_type=self.cost_type)
            for link in path:
                link.aux_flow += od.demand

    def bisection(self, network: Network) -> float:
        def derivative(alpha: float) -> float:
            if self.cost_type == "c":
                return sum(
                    link.get_cost(link.flow + alpha * (link.aux_flow - link.flow)) * (link.aux_flow - link.flow)
                    for link in network.link_set
                )
            return sum(
                link.get_marginal_cost(link.flow + alpha * (link.aux_flow - link.flow))
                * (link.aux_flow - link.flow)
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
