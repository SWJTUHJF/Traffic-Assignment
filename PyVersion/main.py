from g_parser import NetworkParser as NP
from a_path_based import GradientProjection as GP
from a_link_based import FrankWolfe as FW, MSA


if __name__ == "__main__":
    parser = NP()
    # net = parser.load("ChicagoSketch", demand_level=1.0)
    net = parser.load("SiouxFalls", demand_level=1.0)
    solver = MSA()
    solver.run_MSA_UE(net, verbose=True, tol_gap=1e-4)
