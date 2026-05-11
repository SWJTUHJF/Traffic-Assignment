#include "traffic_assignment/link_based.hpp"
#include "traffic_assignment/parser.hpp"
#include "traffic_assignment/shortest_path.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const std::filesystem::path data_root = std::filesystem::current_path() / "data";
        ta::NetworkParser parser(data_root);
        ta::Network network = parser.load("SiouxFalls", 1.0);

        const ta::Path path = ta::shortest_path(network, 0, 23, ta::CostType::Cost);
        const auto nodes = path.node_ids(network.link_set);

        std::cout << "Network: " << network.name << '\n';
        std::cout << "Nodes: " << network.num_node
                  << ", Links: " << network.num_link
                  << ", ODs: " << network.num_od
                  << ", Total flow: " << std::fixed << std::setprecision(1) << network.total_flow << '\n';

        std::cout << "Shortest path from 1 to 24 cost: "
                  << std::fixed << std::setprecision(1) << path.cost << '\n';

        std::cout << "Path nodes: ";
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (i != 0) {
                std::cout << " -> ";
            }
            std::cout << nodes[i] + 1;
        }
        std::cout << '\n';

        std::cout << "Path links:\n";
        for (int link_id : path.link_ids) {
            const auto& link = network.link_set.at(static_cast<std::size_t>(link_id));
            std::cout << "  " << link.tail_id + 1 << " -> " << link.head_id + 1
                      << "  cost=" << std::fixed << std::setprecision(1) << link.cost << '\n';
        }

        std::cout << "\nRunning Frank-Wolfe UE on SiouxFalls...\n";
        ta::FrankWolfe solver;
        solver.run_FW_UE(network, 1e-4, false);
        std::cout << "FW iterations: " << solver.iterations()
                  << ", final gap: " << std::scientific << std::setprecision(6) << solver.current_gap()
                  << ", final TSTT: " << std::fixed << std::setprecision(1) << network.tstt()
                  << '\n';

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
