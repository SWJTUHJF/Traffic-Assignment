from build_network import Network, Path

from time import perf_counter as pc
from typing import Literal


Var = Literal["mc", "c"]


class LinkBased:
    def run_FW_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False):
        start_time, iter_times, cur_gap = pc(), 0, float('inf')
        self.all_or_nothing(network, "c")
        for link in network.link_set:
            link.flow = link.aux_flow
        while cur_gap > tol_gap:
            network.update_all_link_cost()
            self.all_or_nothing(network, "c")
            step = self.bisection(network, "c")
            self.update_flow(step, network)
            cur_gap = RG(network, "c")
            iter_times += 1
            if verbose:
                print(f"Iteration {iter_times}: current gap={cur_gap}")
        print(f"Running time={pc()-start_time:.5f}s, TSTT={network.tstt}")

    def run_FW_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False):
        start_time, iter_times, cur_gap = pc(), 0, float('inf')
        self.all_or_nothing(network, "mc")
        for link in network.link_set:
            link.flow = link.aux_flow
        while cur_gap > tol_gap:
            network.update_all_link_marginal_cost()
            self.all_or_nothing(network, "mc")
            step = self.bisection(network, "mc")
            self.update_flow(step, network)
            cur_gap = RG(network, "mc")
            iter_times += 1
            if verbose:
                print(f"Iteration {iter_times}: current gap={cur_gap}")
        network.update_all_link_cost()
        print(f"Running time={pc()-start_time:.5f}s, TSTT={network.tstt}")
    
    def run_CFW_UE(self):  # conjugate FW
        pass

    def run_CFW_SO(self):
        pass

    def run_BCFW_UE(self):  # bi-conjugate FW
        pass

    def run_BCFW_SO(self):  # bi-conjugate FW
        pass

    def update_flow(self, step: float, network: Network) -> None:
        for link in network.link_set[1:]:
            link.flow = link.flow + step * (link.aux_flow - link.flow)

    def all_or_nothing(self, network: Network, cost_type: Var) -> None:
        for link in network.link_set[1:]:
            link.aux_flow = 0
        for od in network.od_set:
            path, _ = od.shortest_path_and_cost(cost_type=cost_type)
            for link in path:
                link.aux_flow += od.demand

    def bisection(self, network: Network, cost_type: Var) -> float:
        def derivative_f(alpha: float, network: Network, cost_type: Var) -> float:
            if cost_type == "c":
                res = sum([link.get_cost(link.flow + alpha * (link.aux_flow - link.flow)) 
                        * (link.aux_flow - link.flow) for link in network.link_set[1:]])
            else:
                res = sum([link.get_marginal_cost(link.flow + alpha * (link.aux_flow - link.flow)) 
                        * (link.aux_flow - link.flow) for link in network.link_set[1:]])
            return res
        left, right, mid = 0, 1, 0.5
        max_iter_times, iter_times = 30, 1
        while iter_times < max_iter_times:
            if derivative_f(mid, network, cost_type) * derivative_f(right, network, cost_type) > 0:
                right = mid
            else:
                left = mid
            mid = (right + left) / 2
            iter_times += 1
        return mid


class PathBased:
    def run_GP_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False):
        start_time = pc()
        # 1 initialize working set
        for od in network.od_set:
            path = Path(od.origin, od.destination, od.shortest_path_and_cost(cost_type="c")[0])
            path.add_flow(od.demand)
            od.working_set.append(path)
            for link in path.included_links:
                link.update_cost()
        # 2 Main loop: shift flow
        iter_times, cur_gap = 0, float('inf')
        while cur_gap > tol_gap:
            for od in network.od_set:
                # 2a find the shortest path and add it to the working set if it's not in there
                basic_path, min_dist = od.shortest_path_and_cost(cost_type="c")
                for path in od.working_set:
                    if path.included_links == basic_path:
                        basic_path = path
                        break
                else:
                    basic_path = Path(od.origin, od.destination, basic_path)
                    od.working_set.append(basic_path)
                # 2b shift flow
                for non_basic_path in od.working_set:
                    if non_basic_path == basic_path:
                        continue
                    xor_links = set(non_basic_path.included_links) ^ set(basic_path.included_links)
                    denominator = sum([link.cost_function_derivative for link in xor_links])
                    shifted_flow = min((non_basic_path.cost - min_dist) / denominator, non_basic_path.flow)
                    non_basic_path.add_flow(-shifted_flow)
                    basic_path.add_flow(shifted_flow)
                    for link in xor_links:
                        link.update_cost()
                # 3 drop the unused working path
                od.working_set = [path for path in od.working_set if path.flow > 0]
            # 4 convergence check
            cur_gap = RG(network, cost_type="c")
            iter_times += 1
            if verbose:
                print(f'Iteration {iter_times}: current gap = {cur_gap}')
        print(f"Running time = {pc() - start_time:.5f}, TSTT = {network.tstt}")

    def run_GP_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False):
        start_time = pc()
        # 1 initialize working set
        for od in network.od_set:
            path = Path(od.origin, od.destination, od.shortest_path_and_cost(cost_type="mc")[0])
            path.add_flow(od.demand)
            od.working_set.append(path)
            for link in path.included_links:
                link.update_marginal_cost()
        # 2 Main loop: shift flow
        iter_times, cur_gap = 0, float('inf')
        while cur_gap > tol_gap:
            for od in network.od_set:
                # 2a find the shortest path and add it to the working path set if it's not in there
                basic_path, min_dist = od.shortest_path_and_cost(cost_type="mc")
                for working_path in od.working_set:
                    if working_path.included_links == basic_path:
                        basic_path = working_path
                        break
                else:
                    basic_path = Path(od.origin, od.destination, basic_path)
                    od.working_set.append(basic_path)
                # 2b shift flow
                for working_path in od.working_set:
                    if working_path == basic_path:
                        continue
                    xor_links = set(working_path.included_links) ^ set(basic_path.included_links)
                    temp = sum([link.marginal_cost_function_derivative for link in xor_links])
                    shifted_flow = min((working_path.marginal_cost - min_dist) / temp, working_path.flow)
                    working_path.add_flow(-shifted_flow)
                    basic_path.add_flow(shifted_flow)
                    for link in xor_links:
                        link.update_marginal_cost()
                # 3 drop the unused working path
                od.working_set = [path for path in od.working_set if path.flow > 0]
            # 4 convergence check
            cur_gap = RG(network, cost_type="mc")
            iter_times += 1
            if verbose:
                print(f'Iteration {iter_times}: current gap = {cur_gap}')
        network.update_all_link_cost()
        print(f"Running time = {pc() - start_time:.5f}, TSTT = {network.tstt}")
    
    def run_MS_UE(self):  # Manifold suboptimization
        pass

    def run_MS_SO(self):  # Manifold suboptimization
        pass


class BushBased:
    pass



def RG(network: Network, cost_type: Var = "c") -> float:  # classic relative gap
    denominator = 0
    for od in network.od_set:
        _, min_dist = od.shortest_path_and_cost(cost_type=cost_type)
        denominator += min_dist * od.demand
    numerator = network.tstt if cost_type == "c" else sum([link.flow * link.marginal_cost for link in network.link_set[1:]])
    return numerator / denominator - 1


def AEC(network):  # average excess cost
    pass


def MEC(network):  # maximum excess cost
    pass


if __name__ == "__main__":
    sf = Network(name="SiouxFalls")
    solver = PathBased()
    solver.run_GP_UE(sf, tol_gap=1e-4, verbose=True)