from g_parser import NetworkParser as NP
from a_path_based import GradientProjection as GP
from a_link_based import FrankWolfe as FW, MSA
from a_bush_based import DBA, BBA, NBA, QBA


if __name__ == "__main__":
    parser = NP()
    # net = parser.load("ChicagoSketch", demand_level=1.0)
    net = parser.load("SiouxFalls", demand_level=1.0)
    # solver = GP()
    # solver.run_GP_UE(net, verbose=True, tol_gap=1e-4)
    solver = DBA()
    solver.run_DBA_UE(net, verbose=True, tol_gap=1e-4)
