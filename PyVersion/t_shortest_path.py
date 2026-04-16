from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from PyVersion.graph.parser import NetworkParser


if __name__ == "__main__":
    parser = NetworkParser()
    net = parser.load("SiouxFalls", demand_level=1.0)
    node_1, node_24 = net.node_set[0], net.node_set[23]
    print(net.shortest_path(node_1, node_24, cost_type="c"))
