from a_base_solver import BaseSolver
from g_network import Network


class DBA(BaseSolver):  # Dial 2006
    def run_DBA_UE(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="c", tol_gap=tol_gap, verbose=verbose)

    def run_DBA_SO(self, network: Network, tol_gap: float = 1e-4, verbose: bool = False) -> None:
        self.solve(network, cost_type="mc", tol_gap=tol_gap, verbose=verbose)
        
    def initialize(self) -> None:
        raise NotImplementedError

    def main_loop_step(self) -> None:
        raise NotImplementedError


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
