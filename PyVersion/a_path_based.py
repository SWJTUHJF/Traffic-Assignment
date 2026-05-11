from a_base_solver import BaseSolver
from g_network import Network


class GradientProjection(BaseSolver):
    def run_GP_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, kind="tt", tol_gap=tol_gap, verbose=verbose)

    def run_GP_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, kind="mtt", tol_gap=tol_gap, verbose=verbose)
    
    def preprocess(self) -> None:
        super().preprocess()
        self.network.reset_assignment()

    def initialize(self) -> None:
        for od in self.network.od_set:
            basic_path = od.shortest_path(self.kind)
            basic_path.add_flow(od.demand)
            od.working_set = [basic_path]

            for link in basic_path:
                link.update_cost(self.kind)

    def main_loop_step(self) -> None:
        for od in self.network.od_set:
            basic_path = od.shortest_path(self.kind)
            for path in od.working_set:
                if path == basic_path:
                    basic_path = path
                    break
            else:
                od.working_set.append(basic_path)
            min_dist = basic_path.get_cost(self.kind)

            for non_basic_path in od.working_set:
                if non_basic_path == basic_path:
                    continue

                xor_links = set(non_basic_path.included_links) ^ set(basic_path.included_links)

                denominator = sum(link.get_cost_derivative(self.kind) for link in xor_links)
                numerator = non_basic_path.get_cost(self.kind) - min_dist
                shifted_flow = min(numerator / denominator, non_basic_path.flow)

                non_basic_path.add_flow(-shifted_flow)
                basic_path.add_flow(shifted_flow)

                for link in xor_links:
                    link.update_cost(self.kind)

            od.working_set = [path for path in od.working_set if path.flow > 1e-12]


class ManifoldSuboptimization(BaseSolver):
    pass
