from __future__ import annotations

from .alg_base_solver import BaseSolver
from ..graph.network import Network, Path


class PathBased(BaseSolver):
    def run_GP_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_GP_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)

    def run_MS_UE(self) -> None:
        raise NotImplementedError

    def run_MS_SO(self) -> None:
        raise NotImplementedError

    def initialize(self) -> None:
        assert self.network is not None

        for od in self.network.od_set:
            basic_path = od.shortest_path(self.cost_type)
            path = Path(od.origin, od.destination, basic_path.included_links)
            path.add_flow(od.demand)
            od.working_set = [path]

            for link in path.included_links:
                if self.cost_type == "c":
                    link.update_cost()
                else:
                    link.update_marginal_cost()

    def main_loop_step(self) -> None:
        assert self.network is not None

        for od in self.network.od_set:
            basic_path = od.shortest_path(self.cost_type)
            basic_links = basic_path.included_links
            min_dist = basic_path.cost if self.cost_type == "c" else basic_path.marginal_cost

            basic_working_path = None
            for candidate in od.working_set:
                if candidate.included_links == basic_links:
                    basic_working_path = candidate
                    break

            if basic_working_path is None:
                basic_working_path = Path(od.origin, od.destination, basic_links)
                od.working_set.append(basic_working_path)

            for non_basic_path in list(od.working_set):
                if non_basic_path == basic_working_path:
                    continue

                xor_links = set(non_basic_path.included_links) ^ set(basic_working_path.included_links)
                if not xor_links:
                    continue

                if self.cost_type == "c":
                    denominator = sum(link.d_cost for link in xor_links)
                    numerator = non_basic_path.cost - min_dist
                else:
                    denominator = sum(link.d_marginal_cost for link in xor_links)
                    numerator = non_basic_path.marginal_cost - min_dist

                if denominator <= 0.0 or numerator <= 0.0:
                    continue

                shifted_flow = min(numerator / denominator, non_basic_path.flow)
                if shifted_flow <= 0.0:
                    continue

                non_basic_path.add_flow(-shifted_flow)
                basic_working_path.add_flow(shifted_flow)

                for link in xor_links:
                    if self.cost_type == "c":
                        link.update_cost()
                    else:
                        link.update_marginal_cost()

            od.working_set = [path for path in od.working_set if path.flow > 1e-12]

    def postprocess(self) -> None:
        assert self.network is not None
        if self.cost_type == "mc":
            self.network.update_all_link_cost()
