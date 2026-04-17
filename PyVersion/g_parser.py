from pathlib import Path
import re

from g_network import Network, Node


class NetworkParser:
    def __init__(self, data_root: str | Path | None = None):
        if data_root is None:
            self.data_root = Path(__file__).resolve().parent / "data"
        else:
            self.data_root = Path(data_root)

    def load(self, name: str, demand_level: float = 1.0) -> Network:
        network = Network(name=name, demand_level=demand_level)

        net_file = self._resolve_file(name, f"{name}_net.txt")
        trips_file = self._resolve_file(name, f"{name}_trips.txt")

        self._read_network(network, net_file)
        self._read_trips(network, trips_file)
        network.update_all_link_cost_and_marginal_cost()

        return network

    def _resolve_file(self, name: str, filename: str) -> Path:
        p = self.data_root / name / filename
        if p.exists():
            return p
        raise FileNotFoundError(f"Cannot find {filename} under {self.data_root / name}")

    def _read_network(self, network: Network, file_path: Path) -> None:
        with open(file_path) as f:
            lines = f.readlines()

        pattern = re.compile(r"[0-9A-Za-z.~]+")
        lines = [pattern.findall(line) for line in lines if pattern.findall(line) != []]

        num_node, num_link = None, None
        for i in range(len(lines)):
            line = lines[i]
            if "NUMBER" in line and "NODES" in line:
                num_node = int(line[-1])
            if "NUMBER" in line and "LINKS" in line:
                num_link = int(line[-1])
            if "~" in line and "capacity" in line:
                lines = lines[i + 1:]
                break
        if num_node is None or num_link is None:
            raise ValueError(f"Cannot parse number of nodes and links from {file_path}")

        network.node_set = [Node(i) for i in range(num_node)]
        for line in lines:
            network.add_link(
                tail_id=int(float(line[0]) - 1),
                head_id=int(float(line[1]) - 1),
                capacity=float(line[2]),
                length=float(line[3]),
                free_flow_time=float(line[4]),
                alpha=float(line[5]),
                beta=float(line[6]),
            )

        if network.num_link != num_link:
            raise ValueError(f"Expected {num_link} links, parsed {network.num_link} in {file_path}")

    def _read_trips(self, network: Network, file_path: Path) -> None:
        with open(file_path) as f:
            lines = f.readlines()

            total_flow = None
            pattern = re.compile(r"[a-zA-Z0-9.]+")
            lines = [pattern.findall(line) for line in lines if pattern.findall(line) != []]
            for i, line in enumerate(lines):
                if "TOTAL" in line:
                    total_flow = float(line[-1])
                if "Origin" in line:
                    lines = lines[i:]
                    break

            origin_id, parsed_demand = None, 0.0
            for line in lines:
                if "Origin" in line:
                    origin_id = int(line[-1])
                else:
                    for i in range(len(line) // 2):
                        destination_id = int(line[2 * i])
                        demand = float(line[2 * i + 1]) * network.demand_level
                        parsed_demand += demand
                        if demand != 0.0 and origin_id is not None:
                            network.add_od(origin_id - 1, destination_id - 1, demand)

        if total_flow is not None:
            expected = total_flow * network.demand_level
            if expected > 0.0 and abs(parsed_demand - expected) / expected > 0.01:
                raise ValueError(f"Inconsistent demand in {file_path}: parsed={parsed_demand}, expected={expected}")

        network.total_flow = sum(od.demand for od in network.od_set)