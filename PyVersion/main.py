from g_parser import NetworkParser as NP


if __name__ == "__main__":
    parser = NP()
    net = parser.load("SiouxFalls", demand_level=1.0)
    node_1, node_24 = net.node_set[0], net.node_set[23]
    print(net.shortest_path(node_1, node_24))
