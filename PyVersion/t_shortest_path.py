from g_parser import NetworkParser


parser = NetworkParser()
net = parser.load("SiouxFalls", demand_level=1.0)
print(net.shortest_path(1, 24, cost_type="tt"))
