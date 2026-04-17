from a_base_solver import BaseSolver
from g_network import Network


class DBA(BaseSolver):  # Dial 2006
    def run_DBA_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_DBA_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)
        
    def initialize(self) -> None:
        self.network.construct_bushes()

        for bush in self.network.bushes:
            search_result = bush.search_sp(cost_type=self.cost_type, pre_terminate=False)
            bush.tree_links = [link for link in search_result.prev_link.values() if link is not None]
            for od in bush.included_ods:
                for link in search_result.path_to(od.destination):
                    link.flow += od.demand

        if self.cost_type == "c":
            self.network.update_all_link_cost()
        else:
            self.network.update_all_link_marginal_cost()

    def main_loop_step(self) -> None:
        self.expand_bushes()
        self.update_bushes_flow()
        self.remove_unused_links()
    
    def expand_bushes(self) -> None:
        for bush in self.network.bushes:
            search_result = bush.search_sp(cost_type=self.cost_type, pre_terminate=False)
            for link in self.network.link_set:
                tail_dist, head_dist = search_result.dist[link.tail], search_result.dist[link.head]
                edge_cost = link.cost if self.cost_type == "c" else link.marginal_cost
                if link not in bush.tree_links and tail_dist + edge_cost < head_dist:
                    bush.tree_links.append(link)

    def update_bushes_flow(self) -> None:
        
        for bush in self.network.bushes:
            # 1 Ascending pass
            bush.ascending_pass(cost_type=self.cost_type)
            # 2 Convergence test
            if bush.max_dist_diff() < self.tol_gap:
                continue
            # 3 Descending pass
    
    def remove_unused_links(self) -> None:
        for bush in self.network.bushes:
            for link in bush.tree_links:
                if link.flow == 0:
                    bush.tree_links.remove(link)


class BBA(BaseSolver):  # Bar-Gera 2002
    def run_BBA_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_BBA_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)
        
    def initialize(self) -> None:
        raise NotImplementedError

    def main_loop_step(self) -> None:
        raise NotImplementedError


class NBA(BaseSolver):  # Nie 2007
    def run_NBA_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_NBA_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)
        
    def initialize(self) -> None:
        raise NotImplementedError

    def main_loop_step(self) -> None:
        raise NotImplementedError


class QBA(BaseSolver):  # Nie 2010
    def run_QBA_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_QBA_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)
        
    def initialize(self) -> None:
        raise NotImplementedError

    def main_loop_step(self) -> None:
        raise NotImplementedError
