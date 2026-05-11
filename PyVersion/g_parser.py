from pathlib import Path
import re

from g_network import Network, Node, Link, OD


class NetworkParser:
    def __init__(self, data_root: str | Path | None = None):
        self.data_root = Path(data_root) if data_root else Path(__file__).resolve().parent / "data"

    def load(self, name: str, demand_level: float = 1.0) -> Network:
        network = Network(name=name, demand_level=demand_level)

        net_file = self._resolve_file(name, f"{name}_net.txt")
        trips_file = self._resolve_file(name, f"{name}_trips.txt")

        self._read_network(network, net_file)
        self._read_trips(network, trips_file)

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

        num_link = None
        for i in range(len(lines)):
            line = lines[i]
            if "NUMBER" in line and "NODES" in line:
                network.num_node = int(line[-1])
            if "NUMBER" in line and "LINKS" in line:
                num_link = int(line[-1])
            if "~" in line and "capacity" in line:
                lines = lines[i + 1:]
                break
        if network.num_node is None or num_link is None:
            raise ValueError(f"Cannot parse number of nodes and links from {file_path}")
        # create node set
        network.node_set = [Node(i) for i in range(network.num_node)]
        # create link set
        for line in lines:
            tail = network.node_set[int(float(line[0]) - 1)]
            head = network.node_set[int(float(line[1]) - 1)]
            link = Link(
                link_id=network.num_link,
                tail=tail,
                head=head,
                capacity=float(line[2]),
                length=float(line[3]),
                free_flow_time=float(line[4]),
                alpha=float(line[5]),
                beta=float(line[6]),
            )
            network.link_set.append(link)
            tail.link_out.append(link)
            head.link_in.append(link)
            network.num_link += 1
        if network.num_link != num_link:
            print(f"Warning: Expected {num_link} links, parsed {network.num_link} in {file_path}")

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
                continue

            if not origin_id:
                raise ValueError("Origin not detected")
            for i in range(len(line) // 2):
                destination_id = int(line[2 * i])
                demand = float(line[2 * i + 1]) * network.demand_level
                if demand <= 0.0 or origin_id == destination_id:
                    continue
                
                od = OD(
                    origin=network.node_set[origin_id - 1],
                    destination=network.node_set[destination_id - 1],
                    demand=demand,
                    network=network,
                    )
                network.od_set.append(od)
                network.num_od += 1
                parsed_demand += demand

        if total_flow is not None:
            expected = total_flow * network.demand_level
            if expected > 0.0 and abs(parsed_demand - expected) / expected > 0.01:
                raise ValueError(f"Inconsistent demand in {file_path}: parsed={parsed_demand}, expected={expected}")

        network.total_flow = sum(od.demand for od in network.od_set)
