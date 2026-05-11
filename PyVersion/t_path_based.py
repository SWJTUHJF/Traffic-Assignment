from a_path_based import GradientProjection as GP
from g_parser import NetworkParser


parser = NetworkParser()
network = parser.load("SiouxFalls", demand_level=1.0)
solver = GP()
solver.run_GP_UE(network, verbose=True, tol_gap=1e-4)
solver.run_GP_SO(network, verbose=True, tol_gap=1e-4)
