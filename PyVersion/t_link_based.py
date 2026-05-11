from a_link_based import FrankWolfe, MSA
from g_parser import NetworkParser


parser = NetworkParser()
network = parser.load("SiouxFalls", demand_level=1.0)
# solver = FrankWolfe()
# solver.run_FW_UE(network, verbose=True, tol_gap=1e-4)
# solver.run_FW_SO(network, verbose=True, tol_gap=1e-4)

solver = MSA()
solver.run_MSA_UE(network, verbose=True, tol_gap=1e-4)
# solver.run_MSA_SO(network, verbose=True, tol_gap=1e-4)
