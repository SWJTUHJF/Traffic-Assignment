from re import S

from g_sp import SearchResult, nodes_from_links
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
            bush.tree_links = {link: 0 for link in search_result.prev_link.values() if link is not None}
            for od in bush.included_ods:
                for link in search_result.path_to(od.destination):
                    bush.add_flow(link, od.demand)

        if self.cost_type == "c":
            self.network.update_all_link_cost()
        else:
            self.network.update_all_link_marginal_cost()

    def main_loop_step(self) -> None:
        self.expand_bushes()
        self.update_bushes_flow()
        self.remove_unused_links()
    
    # def expand_bushes(self) -> None:
    #     for bush in self.network.bushes:
    #         search_result = bush.search_sp(cost_type=self.cost_type, pre_terminate=False, global_sp=False)
    #         for link in self.network.link_set:
    #             tail_dist, head_dist = search_result.dist[link.tail], search_result.dist[link.head]
    #             edge_cost = link.cost if self.cost_type == "c" else link.marginal_cost
    #             if link not in bush.tree_links and tail_dist + edge_cost < head_dist:
    #                 bush.tree_links[link] = 0

    # expand_bushes
    def expand_bushes(self) -> None:
        for bush in self.network.bushes:
            bush.update_ascending_pass(cost_type=self.cost_type)
            for link in self.network.link_set:
                if link in bush.tree_links:
                    continue

                c = link.cost if self.cost_type == "c" else link.marginal_cost
                u_i = bush.min_dist[link.tail]
                u_j = bush.min_dist[link.head]
                U_i = bush.max_dist[link.tail]
                U_j = bush.max_dist[link.head]

                if u_i == float("inf") or u_j == float("inf"):
                    continue
                if U_i == float("-inf") or U_j == float("-inf"):
                    continue

                if u_i + c < u_j and U_i + c < U_j:
                    bush.tree_links[link] = 0.0


    def update_bushes_flow(self) -> None:
        for bush in self.network.bushes:
            # 1 Ascending pass
            bush.update_ascending_pass(cost_type=self.cost_type)

            # 2 Convergence test
            if bush.max_dist_diff() < self.tol_gap:
                continue
            # 3 Descending pass
            d_topo_order = bush.topo_order[::-1]
            for node_j in d_topo_order:
                if node_j is bush.origin:
                    continue
                if bush.min_pred[node_j] is None or bush.max_pred[node_j] is None:
                    continue

                search_result_sp = SearchResult(bush.origin, bush.min_dist, bush.min_pred)
                search_result_lp = SearchResult(bush.origin, bush.max_dist, bush.max_pred)
                sp, lp = search_result_sp.path_to(node_j), search_result_lp.path_to(node_j)
                s_nodes, l_nodes = nodes_from_links(bush.origin, sp), nodes_from_links(bush.origin, lp)

                common_idx = 0
                common_len = min(len(s_nodes), len(l_nodes))
                while common_idx + 1 < common_len and s_nodes[common_idx + 1] is l_nodes[common_idx + 1]:
                    common_idx += 1

                node_i = s_nodes[common_idx]
                if node_i is node_j:
                    continue

                # p_i and P_i are the common shortest/longest prefixes from origin to i.
                p_i = sp[:common_idx]
                P_i = lp[:common_idx]
                shortest_j = sp[common_idx:]
                longest_j = lp[common_idx:]

                # g = (bush.max_dist[node_j] - bush.max_dist[node_i]) - (bush.min_dist[node_j] - bush.min_dist[node_i])
                short_cost = sum(link.cost for link in shortest_j)
                long_cost = sum(link.cost for link in longest_j)
                g = long_cost - short_cost

                union_links = set(shortest_j) | set(longest_j)
                h = sum(link.d_cost if self.cost_type == "c" else link.d_marginal_cost for link in union_links)
                dx = min(g / h, min(bush.tree_links[link] for link in longest_j))

                for link in shortest_j:
                    bush.add_flow(link, dx)
                for link in longest_j:
                    bush.add_flow(link, -dx)
                self.network.update_all_link_cost() if self.cost_type == "c" else self.network.update_all_link_marginal_cost()
    
    def remove_unused_links(self) -> None:
        for bush in self.network.bushes:
            bush.tree_links = {link: flow for link, flow in bush.tree_links.items() if flow > 0}


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
