from a_base_solver import BaseSolver
from g_network import Network


class GradientProjection(BaseSolver):
    def run_GP_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_GP_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)

    def initialize(self) -> None:
        for od in self.network.od_set:
            basic_path = od.shortest_path(self.cost_type)
            basic_path.add_flow(od.demand)
            od.working_set = [basic_path]

            for link in basic_path:
                if self.cost_type == "c":
                    link.update_cost()
                else:
                    link.update_marginal_cost()

    def main_loop_step(self) -> None:
        for od in self.network.od_set:
            basic_path = od.shortest_path(self.cost_type)
            for path in od.working_set:
                if path == basic_path:
                    basic_path = path
                    break
            else:
                od.working_set.append(basic_path)
            min_dist = basic_path.cost if self.cost_type == "c" else basic_path.marginal_cost

            for non_basic_path in od.working_set:
                if non_basic_path == basic_path:
                    continue

                xor_links = set(non_basic_path.included_links) ^ set(basic_path.included_links)

                if self.cost_type == "c":
                    denominator = sum(link.d_cost for link in xor_links)
                    numerator = non_basic_path.cost - min_dist
                else:
                    denominator = sum(link.d_marginal_cost for link in xor_links)
                    numerator = non_basic_path.marginal_cost - min_dist

                shifted_flow = min(numerator / denominator, non_basic_path.flow)

                non_basic_path.add_flow(-shifted_flow)
                basic_path.add_flow(shifted_flow)

                for link in xor_links:
                    if self.cost_type == "c":
                        link.update_cost()
                    else:
                        link.update_marginal_cost()

            od.working_set = [path for path in od.working_set if path.flow > 1e-12]


class ManifoldSuboptimization(BaseSolver):
    def run_MS_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_MS_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)

    def initialize(self) -> None:
        for od in self.network.od_set:
            basic_path = od.shortest_path(self.cost_type)
            basic_path.add_flow(od.demand)
            od.working_set = [basic_path]

            for link in basic_path:
                if self.cost_type == "c":
                    link.update_cost()
                else:
                    link.update_marginal_cost()

    def main_loop_step(self) -> None:
        for od in self.network.od_set:
            basic_path = od.shortest_path(self.cost_type)
            for path in od.working_set:
                if path == basic_path:
                    basic_path = path
                    break
            else:
                od.working_set.append(basic_path)
            min_dist = basic_path.cost if self.cost_type == "c" else basic_path.marginal_cost

            for non_basic_path in od.working_set:
                if non_basic_path == basic_path:
                    continue

                xor_links = set(non_basic_path.included_links) ^ set(basic_path.included_links)

                if self.cost_type == "c":
                    denominator = sum(link.d_cost for link in xor_links)
                    numerator = non_basic_path.cost - min_dist
                else:
                    denominator = sum(link.d_marginal_cost for link in xor_links)
                    numerator = non_basic_path.marginal_cost - min_dist

                shifted_flow = min(numerator / denominator, non_basic_path.flow)

                non_basic_path.add_flow(-shifted_flow)
                basic_path.add_flow(shifted_flow)

                for link in xor_links:
                    if self.cost_type == "c":
                        link.update_cost()
                    else:
                        link.update_marginal_cost()

            od.working_set = [path for path in od.working_set if path.flow > 1e-12]
