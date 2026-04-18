from re import S

from g_sp import SearchResult, nodes_from_links
from a_base_solver import BaseSolver
from g_network import Network, Bush, Link, Node


class DBA(BaseSolver):  # Dial 2006
    def run_DBA_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_DBA_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)
        
    def initialize(self) -> None:
        self.network.reset_assignment()
        self.network.construct_bushes()
        for bush in self.network.bushes:
            bush.initialize(self.cost_type)

    def main_loop_step(self) -> None:
        for bush in self.network.bushes:
            bush.expand(self.cost_type)
            self.update_bushes_flow(bush)
            bush.remove_unused_links()

    def update_bushes_flow(self, bush: Bush, epsilon: float = 1e-12) -> None:
        def path_from_pred(pred: dict[Node, Link | None], destination: Node) -> list[Link]:
            links: list[Link] = []
            node = destination
            while node is not bush.origin:
                link = pred[node]
                if link is None:
                    return []
                links.append(link)
                node = link.tail
            links.reverse()
            return links

        # Step 0: update current costs on links in the bush
        for link in bush.tree_links:
            if self.cost_type == "c":
                link.update_cost()
            else:
                link.update_marginal_cost()

        # Step 1: ascending pass
        bush.update_ascending_pass(cost_type=self.cost_type)

        # Step 2: convergence test for this bush
        if bush.max_dist_diff() < epsilon:
            return

        # Step 3: descending pass
        for node_j in reversed(bush.topo_order):
            if node_j is bush.origin:
                continue
            longest_path = path_from_pred(bush.max_pred, node_j)
            shortest_path = path_from_pred(bush.min_pred, node_j)

            if longest_path == shortest_path:
                continue

            longest_nodes = nodes_from_links(longest_path)
            shortest_nodes = nodes_from_links(shortest_path)

            common_node = None
            for node in longest_nodes[:-1]:
                if node in shortest_nodes:
                    common_node = node
            if common_node is None:
                raise ValueError(f"No common node found in longest and shortest paths")
            
            # common_node = bush.origin
            # for i in range(min(len(longest_nodes), len(shortest_nodes))):
            #     if longest_nodes[i] == shortest_nodes[i]:
            #         common_node = longest_nodes[i]
            #     else:
            #         break
            
            for link_id, link in enumerate(longest_path):
                if link.tail == common_node:
                    longest_path = longest_path[link_id:]
                    break
            
            for link_id, link in enumerate(shortest_path):
                if link.tail == common_node:
                    shortest_path = shortest_path[link_id:]
                    break

            numerator = (bush.max_dist[node_j] - bush.max_dist[common_node]) - (bush.min_dist[node_j] - bush.min_dist[common_node])
            xor_link = set(longest_path) | set(shortest_path)

            if self.cost_type == "c":
                denominator = sum(link.d_cost for link in xor_link)
            else:
                denominator = sum(link.d_marginal_cost for link in xor_link)
            # if denominator < 0.0:
            #     raise ValueError(f"Non-positive denominator encountered in flow update: {denominator}")

            # longest segment flow bound must use bush-specific flow
            considered_links = [link for link in longest_path if link not in shortest_path]
            if not considered_links:
                continue
            else:
                max_shift = min(bush.tree_links[link] for link in considered_links)
            if max_shift < 0.0:
                raise ValueError(f"Negative max_shift encountered: {max_shift}")

            dx = min(numerator / denominator, max_shift)
            if dx < 0.0:
                raise ValueError(f"Negative flow shift computed: {dx}")

            # Step 3.2: update flows
            for link in shortest_path:
                bush.add_flow(link, dx)

            for link in longest_path:
                bush.add_flow(link, -dx)

            # Step 3.3: update costs on affected links
            affected_links = set(shortest_path) | set(longest_path)
            for link in affected_links:
                if self.cost_type == "c":
                    link.update_cost()
                else:
                    link.update_marginal_cost()

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
