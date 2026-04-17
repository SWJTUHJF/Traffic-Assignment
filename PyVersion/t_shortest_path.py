from g_parser import NetworkParser


if __name__ == "__main__":
    parser = NetworkParser()
    net = parser.load("SiouxFalls", demand_level=1.0)
    print(net.shortest_path(1, 23, cost_type="c"))
