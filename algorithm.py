"""
- FrankWolfeUserEquilibrium
- FrankWolfeSystemOptimal
- GradientProjectionUserEquilibrium
- GradientProjectionSystemOptimal
"""
from load_network import Network, Path, Link
from utility import spp_algorithm_list, obtain_path_cost, obtain_path_marginal_cost

from copy import deepcopy
import time
from typing import Callable, Literal
from math import inf


__all__ = ["solve"]


def solve(network_name: str = "SiouxFalls",
          TA_type: Literal["SO", "UE"] = "UE",
          main_algorithm: Literal["FW", "GP", "MSA"] = "GP",
          spp_algorithm: Literal["GLC", "LC", "LS", "LSF"] = "LS",
          bisection_gap: float = 1e-4,
          main_gap: float = 1e-4,
          verbose: bool = False):
    # all the models
    network = Network(name=network_name)
    model_dict = {"UE": {"FW": FrankWolfeUserEquilibrium(network=network, algorithm=spp_algorithm,
                                                         bisection_gap=bisection_gap, main_gap=main_gap,
                                                         verbose=verbose),
                         "GP": GradientProjectionUserEquilibrium(network=network, algorithm=spp_algorithm,
                                                                 main_gap=main_gap, verbose=verbose),
                         "MSA": MSAUserEquilibrium(network=network, algorithm=spp_algorithm,
                                                          main_gap=main_gap, verbose=verbose)},
                  "SO": {"FW": FrankWolfeSystemOptimal(network=network, algorithm="LS_SO",
                                                         bisection_gap=bisection_gap, main_gap=main_gap,
                                                       verbose=verbose),
                         "GP": GradientProjectionSystemOptimal(network=network, algorithm="LS_SO",
                                                                 main_gap=main_gap, verbose=verbose)},
                  }
    # run the model
    current_model = model_dict[TA_type][main_algorithm]
    current_model.run()



class FrankWolfeUserEquilibrium:
    def __init__(self, network, algorithm, bisection_gap, main_gap, verbose=False):
        self.network: Network = network
        self.ssp_algorithm: Callable[[Network, int, int], list[Link]] = spp_algorithm_list[algorithm]
        self.bisection_gap: float = bisection_gap  # bisection method convergence
        self.main_gap: float = main_gap  # Frank-Wolfe algorithm convergence
        self.verbose: bool = verbose

    def run(self) -> None:
        s = time.perf_counter()
        print("Initializing...")
        iter_times = 0
        cur_gap = inf
        self.all_or_nothing()
        for link in self.network.link_set:
            link.flow = link.auxiliary_flow
        while cur_gap > self.main_gap:
            self.update_costs()
            self.all_or_nothing()
            step = self.bisection()
            self.update_flow(step)
            cur_gap = self.convergence()
            iter_times += 1
            if self.verbose:
                print("\r", end="")
                print(f"Iteration {iter_times}: Gap={cur_gap}", end="")
        e = time.perf_counter()
        if self.verbose:
            print(f"\nFW for UE finished, time={e-s:.2f}s, tstt={self.network.total_system_travel_time()}")

    def update_costs(self) -> None:
        for link in self.network.link_set[1:]:
            link.update_cost()

    def update_flow(self, step: float) -> None:
        for link in self.network.link_set[1:]:
            link.flow = link.flow + step * (link.auxiliary_flow - link.flow)

    def all_or_nothing(self) -> None:
        for link in self.network.link_set[1:]:
            link.auxiliary_flow = 0
        for od in self.network.od_pair_set:
            ori = od.origin
            dest = od.destination
            demand = od.demand
            shortest_links = self.ssp_algorithm(self.network, ori.node_id, dest.node_id)
            for link in shortest_links:
                link.auxiliary_flow += demand

    def derivative_f(self, alpha) -> float:
        return sum([link.get_specific_cost(link.flow + alpha * (link.auxiliary_flow - link.flow))
                    * (link.auxiliary_flow - link.flow) for link in self.network.link_set[1:]])

    def bisection(self) -> float:
        left, right, mid = 0, 1, 0.5
        max_iter_times = 500
        iter_times = 1
        while abs(self.derivative_f(mid)) > self.bisection_gap:
            iter_times += 1
            if iter_times == max_iter_times:
                raise RuntimeError('Reach maximum iteration times in bisection part but still fail to converge.')
            elif self.derivative_f(mid) * self.derivative_f(right) > 0:
                right = mid
            else:
                left = mid
            mid = (right + left) / 2
        return mid

    def convergence(self) -> float:
        numerator = sum([link.flow * link.cost for link in self.network.link_set[1:]])
        denominator = sum([link.auxiliary_flow * link.cost for link in self.network.link_set[1:]])
        return numerator / denominator - 1


class FrankWolfeSystemOptimal:
    def __init__(self, network, algorithm, bisection_gap, main_gap, verbose=False):
        self.network: Network = network
        self.ssp_algorithm: Callable[[Network, int, int], list[Link]] = spp_algorithm_list[algorithm]
        self.bisection_gap: float = bisection_gap  # bisection method convergence
        self.main_gap: float = main_gap  # Frank-Wolfe algorithm convergence
        self.verbose: bool = verbose

    def run(self) -> None:
        s = time.perf_counter()
        print("Initializing...")
        iter_times = 0
        cur_gap = inf
        self.all_or_nothing()
        for link in self.network.link_set[1:]:
            link.flow = link.auxiliary_flow
        while cur_gap > self.main_gap:
            self.update_costs()
            self.all_or_nothing()
            step = self.bisection()
            self.update_flow(step)
            cur_gap = self.convergence()
            iter_times += 1
            if self.verbose:
                print("\r", end="")
                print(f"Iteration {iter_times}: Gap={cur_gap}", end="")
        e = time.perf_counter()
        if self.verbose:
            print(f"\nFW for SO finished, time={e - s:.2f}s, tstt={self.network.total_system_travel_time()}")

    def update_costs(self) -> None:
        for link in self.network.link_set[1:]:
            link.update_cost()
            link.update_marginal_cost()

    def update_flow(self, step: float) -> None:
        for link in self.network.link_set[1:]:
            link.flow = link.flow + step * (link.auxiliary_flow - link.flow)

    def all_or_nothing(self) -> None:
        for link in self.network.link_set[1:]:
            link.auxiliary_flow = 0
        for od in self.network.od_pair_set:
            ori = od.origin
            dest = od.destination
            demand = od.demand
            shortest_links = self.ssp_algorithm(self.network, ori.node_id, dest.node_id)
            for link in shortest_links:
                link.auxiliary_flow += demand

    def derivative_f(self, alpha) -> float:
        return sum([link.get_specific_marginal_cost(link.flow + alpha * (link.auxiliary_flow - link.flow))
                    * (link.auxiliary_flow - link.flow) for link in self.network.link_set[1:]])

    def bisection(self) -> float:
        left, right, mid = 0, 1, 0.5
        max_iter_times = 500
        iter_times = 1
        while abs(self.derivative_f(mid)) > self.bisection_gap:
            iter_times += 1
            if iter_times == max_iter_times:
                raise RuntimeError('Reach maximum iteration times in bisection part but still fail to converge.')
            elif self.derivative_f(mid) * self.derivative_f(right) > 0:
                right = mid
            else:
                left = mid
            mid = (right + left) / 2
        return mid

    def convergence(self) -> float:
        numerator = sum([link.flow * link.marginal_cost for link in self.network.link_set[1:]])
        denominator = sum([link.auxiliary_flow * link.marginal_cost for link in self.network.link_set[1:]])
        return numerator / denominator - 1


class GradientProjectionUserEquilibrium:
    def __init__(self, network, algorithm, main_gap, verbose=False):
        self.network: Network = network
        self.ssp_algorithm: Callable[[Network, int, int], list] = spp_algorithm_list[algorithm]
        self.main_gap: float = main_gap  # Frank-Wolfe algorithm convergence
        self.verbose: bool = verbose

    def run(self):
        s = time.perf_counter()
        self.initialize_working_path_set()
        iter_times, cur_gap, cur_gap_list = 0, inf, list()
        while cur_gap > self.main_gap:
            for od in self.network.od_pair_set:
                path = self.find_shortest_path_add_to_working_set(od)
                self.shift_flow_between_od(od, path)
                self.drop_unused_path(od)
            cur_gap = self.current_gap()
            iter_times += 1
            if self.verbose:
                print("\r", end="")
                print(f"Iteration {iter_times}: Gap={cur_gap}", end="")
        e = time.perf_counter()
        if self.verbose:
            print(f"\nGP for UE finished, time={e - s:.2f}s, TSTT = {self.network.total_system_travel_time()}")

    def initialize_working_path_set(self):
        for od in self.network.od_pair_set:
            origin, destination, demand = od.origin, od.destination, od.demand
            sp = self.ssp_algorithm(self.network, origin.node_id, destination.node_id)
            path = Path(origin, destination, sp)
            path.path_flow = demand
            od.working_path_set.append(path)
            for link in path.included_links:
                link.flow += demand
        self.network.update_all_link_cost()
        self.network.update_all_working_path_cost()

    def find_shortest_path_add_to_working_set(self, od):
        origin, destination, demand = od.origin, od.destination, od.demand
        sp = self.ssp_algorithm(self.network, origin.node_id, destination.node_id)
        for working_path in od.working_path_set:
            if working_path.included_links == sp:
                path = working_path
                break
        else:
            path = Path(origin, destination, sp)
            od.working_path_set.append(path)
        return path

    def shift_flow_between_od(self, od, path):
        for working_path in od.working_path_set:
            if working_path == path:
                continue
            temp = sum([link.derivative() for link in
                        list(set(working_path.included_links) ^ set(path.included_links))])
            shifted_flow = min((working_path.path_cost - path.path_cost) / temp, working_path.path_flow)
            # update travel times
            working_path.path_flow -= shifted_flow
            for link in working_path.included_links:
                link.flow -= shifted_flow
            path.path_flow += shifted_flow
            for link in path.included_links:
                link.flow += shifted_flow
            self.network.update_all_link_cost()
            self.network.update_all_working_path_cost()

    @staticmethod
    def drop_unused_path(od):
        od.working_path_set = [path for path in od.working_path_set if path.path_flow > 0]

    def current_gap(self):
        SPTT = 0
        for od in self.network.od_pair_set:
            origin, destination, demand = od.origin.node_id, od.destination.node_id, od.demand
            min_path = self.ssp_algorithm(self.network, origin, destination)
            min_dist = obtain_path_cost(min_path)
            SPTT += min_dist * od.demand
        TSTT = sum([link.flow * link.cost for link in self.network.link_set[1:]])
        return (TSTT / SPTT) - 1


class GradientProjectionSystemOptimal:
    def __init__(self, network, algorithm, main_gap, verbose=False):
        self.network: Network = network
        self.ssp_algorithm: Callable[[Network, int, int], list] = spp_algorithm_list[algorithm]
        self.main_gap: float = main_gap  # Frank-Wolfe algorithm convergence
        self.verbose: bool = verbose

    def run(self):
        s = time.perf_counter()
        self.initialize_working_path_set()
        iter_times, cur_gap = 0, inf
        while cur_gap > self.main_gap:
            for od in self.network.od_pair_set:
                path = self.find_shortest_marginal_path_add_to_working_set(od)
                self.shift_flow_between_od(od, path)
                self.drop_unused_path(od)
            cur_gap = self.current_gap()
            iter_times += 1
            if self.verbose:
                print("\r", end="")
                print(f"Iteration {iter_times}: Gap={cur_gap}", end="")
        e = time.perf_counter()
        if self.verbose:
            print(f"\nGP for SO finished, time={e - s:.2f}s, tstt={self.network.total_system_travel_time()}")

    def initialize_working_path_set(self):
        for od_id, od in enumerate(self.network.od_pair_set):
            od.working_path_set = list()
            origin, destination, demand = od.origin, od.destination, od.demand
            sp = self.ssp_algorithm(self.network, origin.node_id, destination.node_id)
            path = Path(origin, destination, sp)
            path.path_flow = demand
            od.working_path_set.append(path)
            for link in path.included_links:
                link.flow += demand
        self.network.update_all_link_marginal_cost()
        self.network.update_all_working_path_marginal_cost()

    def find_shortest_marginal_path_add_to_working_set(self, od):
        origin, destination, demand = od.origin, od.destination, od.demand
        sp = self.ssp_algorithm(self.network, origin.node_id, destination.node_id)
        for working_path in od.working_path_set:
            if working_path.included_links == sp:
                path = working_path
                break
        else:
            path = Path(origin, destination, sp)
            od.working_path_set.append(path)
        return path

    def shift_flow_between_od(self, od, path):
        for working_path in od.working_path_set:
            if working_path == path:
                continue
            temp = sum([link.marginal_cost_function_derivative() for link in
                        list(set(working_path.included_links) ^ set(path.included_links))])
            shifted_flow = min((working_path.path_marginal_cost - path.path_marginal_cost) / temp,
                               working_path.path_flow)
            # update travel times
            working_path.path_flow -= shifted_flow
            for link in working_path.included_links:
                link.flow -= shifted_flow
            path.path_flow += shifted_flow
            for link in path.included_links:
                link.flow += shifted_flow
            self.network.update_all_link_marginal_cost()
            self.network.update_all_working_path_marginal_cost()
            self.network.update_all_link_cost()

    @staticmethod
    def drop_unused_path(od):
        od.working_path_set = [path for path in od.working_path_set if path.path_flow > 0]

    def current_gap(self):
        SPTT = 0
        for od in self.network.od_pair_set:
            origin, destination, demand = od.origin.node_id, od.destination.node_id, od.demand
            min_path = self.ssp_algorithm(self.network, origin, destination)
            min_dist = obtain_path_marginal_cost(min_path)
            SPTT += min_dist * demand
        TSTT = sum([link.flow * link.marginal_cost for link in self.network.link_set[1:]])
        cur_gap = (TSTT / SPTT) - 1
        return cur_gap


class MSAUserEquilibrium:
    def __init__(self, network, algorithm, main_gap, verbose=False):
        self.network: Network = network
        self.ssp_algorithm: Callable[[Network, int, int], list[Link]] = spp_algorithm_list[algorithm]
        self.main_gap: float = main_gap
        self.verbose: bool = verbose

    def run(self) -> None:
        s = time.perf_counter()
        print("Initializing...")
        cur_gap, iter_times = inf, 0
        self.all_or_nothing()
        for link in self.network.link_set:
            link.flow = link.auxiliary_flow
        while cur_gap > self.main_gap:
            iter_times += 1
            self.update_costs()
            self.all_or_nothing()
            step = 1 / (iter_times + 1)
            self.update_flow(step)
            cur_gap = self.convergence()
            if self.verbose:
                print("\r", end="")
                print(f"Iteration {iter_times}: Gap={cur_gap}", end="")
        e = time.perf_counter()
        if self.verbose:
            print(f"\nFW for MSA finished, time={e-s:.2f}s, tstt={self.network.total_system_travel_time()}")

    def update_costs(self) -> None:
        for link in self.network.link_set[1:]:
            link.update_cost()

    def update_flow(self, step: float) -> None:
        for link in self.network.link_set[1:]:
            link.flow = link.flow + step * (link.auxiliary_flow - link.flow)

    def all_or_nothing(self) -> None:
        for link in self.network.link_set[1:]:
            link.auxiliary_flow = 0
        for od in self.network.od_pair_set:
            ori = od.origin
            dest = od.destination
            demand = od.demand
            shortest_links = self.ssp_algorithm(self.network, ori.node_id, dest.node_id)
            for link in shortest_links:
                link.auxiliary_flow += demand

    def convergence(self) -> float:
        numerator = sum([link.flow * link.cost for link in self.network.link_set[1:]])
        denominator = sum([link.auxiliary_flow * link.cost for link in self.network.link_set[1:]])
        return numerator / denominator - 1


"""
1 Find the shortest path based on the flows at last iteration
2 Only update flow after all OD pairs are shifted
3 Consider Bellman's optimality principle
4 Consider parallel computing (less important)
5 Consider the optimal step size to update the flow, as in the Frank-Wolfe algorithm
"""
class GradientProjectionUserEquilibriumJacobi:
    def __init__(self, network, algorithm, main_gap, verbose=False):
        self.network: Network = network
        self.ssp_algorithm: Callable[[Network, int, int], list] = spp_algorithm_list[algorithm]
        self.main_gap: float = main_gap
        self.verbose: bool = verbose

    def run(self):
        s = time.perf_counter()
        self.initialize_working_path_set()
        iter_times, cur_gap, cur_gap_list = 0, inf, list()
        while cur_gap > self.main_gap:
            for od in self.network.od_pair_set:
                copy_net = deepcopy(self.network)
                copy_od = copy_net.od_pair_set[self.network.od_pair_set.index(od)]
                path = self.find_shortest_path_add_to_working_set(copy_od, copy_net)
                self.shift_flow_between_od(copy_od, path, copy_net)
                self.drop_unused_path(od)
                # TODO: finish update_origin_network_path_flow
                self.update_origin_network_path_flow(od.origin.node_id, od.destination.node_id, self.network, copy_net)
            # TODO: finish update_all_link_flows
            self.network.update_all_link_flows()
            self.network.update_all_link_cost()
            self.network.update_all_working_path_cost()
            cur_gap = self.current_gap()
            iter_times += 1
            if self.verbose:
                print("\r", end="")
                print(f"Iteration {iter_times}: Gap={cur_gap}", end="")
        e = time.perf_counter()
        if self.verbose:
            print(f"\nGP for UE finished, time={e - s:.2f}s, TSTT = {self.network.total_system_travel_time()}")

    def initialize_working_path_set(self):
        for od in self.network.od_pair_set:
            origin, destination, demand = od.origin, od.destination, od.demand
            sp = self.ssp_algorithm(self.network, origin.node_id, destination.node_id)
            path = Path(origin, destination, sp)
            path.path_flow = demand
            od.working_path_set.append(path)
            for link in path.included_links:
                link.flow += demand
        self.network.update_all_link_cost()
        self.network.update_all_working_path_cost()

    def find_shortest_path_add_to_working_set(self, od, net):
        origin, destination, demand = od.origin, od.destination, od.demand
        sp = self.ssp_algorithm(net, origin.node_id, destination.node_id)
        for working_path in od.working_path_set:
            if working_path.included_links == sp:
                path = working_path
                break
        else:
            path = Path(origin, destination, sp)
            od.working_path_set.append(path)
        return path

    def shift_flow_between_od(self, od, path, net):
        for working_path in od.working_path_set:
            if working_path == path:
                continue
            temp = sum([link.derivative() for link in
                        list(set(working_path.included_links) ^ set(path.included_links))])
            shifted_flow = min((working_path.path_cost - path.path_cost) / temp, working_path.path_flow)
            # update link and path flow
            working_path.path_flow -= shifted_flow
            for link in working_path.included_links:
                link.flow -= shifted_flow
            path.path_flow += shifted_flow
            for link in path.included_links:
                link.flow += shifted_flow
            net.update_all_link_cost()
            net.update_all_working_path_cost()

    @staticmethod
    def drop_unused_path(od):
        od.working_path_set = [path for path in od.working_path_set if path.path_flow > 0]

    def current_gap(self):
        SPTT = 0
        for od in self.network.od_pair_set:
            origin, destination, demand = od.origin.node_id, od.destination.node_id, od.demand
            min_path = self.ssp_algorithm(self.network, origin, destination)
            min_dist = obtain_path_cost(min_path)
            SPTT += min_dist * od.demand
        TSTT = sum([link.flow * link.cost for link in self.network.link_set[1:]])
        return (TSTT / SPTT) - 1

    def update_origin_network_path_flow(self, o_id, d_id, origin_network, copy_network):
        pass


if __name__ == "__main__":
    sf = Network(name="SiouxFalls")
    model = GradientProjectionUserEquilibriumJacobi(network=sf, algorithm="LS", main_gap=1e-4, verbose=True)
    model.run()
