#include "traffic_assignment/parser.hpp"

#include <cmath>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ta {

namespace {

std::vector<std::string> tokens_from_line(const std::string& line) {
    static const std::regex token_pattern(R"([0-9A-Za-z.~]+)");
    std::vector<std::string> tokens;

    for (std::sregex_iterator it(line.begin(), line.end(), token_pattern), end; it != end; ++it) {
        tokens.push_back(it->str());
    }
    return tokens;
}

std::vector<std::vector<std::string>> read_token_lines(const std::filesystem::path& file_path) {
    std::ifstream input(file_path);
    if (!input) {
        throw std::runtime_error("Cannot open file: " + file_path.string());
    }

    std::vector<std::vector<std::string>> lines;
    std::string line;
    while (std::getline(input, line)) {
        auto tokens = tokens_from_line(line);
        if (!tokens.empty()) {
            lines.push_back(std::move(tokens));
        }
    }
    return lines;
}

bool contains(const std::vector<std::string>& line, const std::string& token) {
    for (const auto& item : line) {
        if (item == token) {
            return true;
        }
    }
    return false;
}

}  // namespace

NetworkParser::NetworkParser(std::filesystem::path data_root)
    : data_root_(std::move(data_root)) {}

Network NetworkParser::load(const std::string& name, double demand_level) const {
    Network network(name, demand_level);

    read_network(network, resolve_file(name, name + "_net.txt"));
    read_trips(network, resolve_file(name, name + "_trips.txt"));
    network.update_all_link_cost_and_marginal_cost();

    return network;
}

std::filesystem::path NetworkParser::resolve_file(const std::string& name, const std::string& filename) const {
    auto path = data_root_ / name / filename;
    if (std::filesystem::exists(path)) {
        return path;
    }
    throw std::runtime_error("Cannot find " + filename + " under " + (data_root_ / name).string());
}

void NetworkParser::read_network(Network& network, const std::filesystem::path& file_path) const {
    auto lines = read_token_lines(file_path);

    int expected_links = -1;
    std::size_t first_data_line = lines.size();

    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto& line = lines[i];
        if (contains(line, "NUMBER") && contains(line, "NODES")) {
            network.resize_nodes(std::stoi(line.back()));
        }
        if (contains(line, "NUMBER") && contains(line, "LINKS")) {
            expected_links = std::stoi(line.back());
        }
        if (contains(line, "~") && contains(line, "capacity")) {
            first_data_line = i + 1;
            break;
        }
    }

    if (network.num_node == 0 || expected_links < 0 || first_data_line == lines.size()) {
        throw std::runtime_error("Cannot parse network metadata from " + file_path.string());
    }

    for (std::size_t i = first_data_line; i < lines.size(); ++i) {
        const auto& line = lines[i];
        if (line.size() < 7) {
            continue;
        }

        network.add_link(
            static_cast<int>(std::stod(line[0])) - 1,
            static_cast<int>(std::stod(line[1])) - 1,
            std::stod(line[2]),
            std::stod(line[3]),
            std::stod(line[4]),
            std::stod(line[5]),
            std::stod(line[6])
        );
    }

    if (network.num_link != expected_links) {
        throw std::runtime_error(
            "Parsed link count mismatch in " + file_path.string() +
            ": parsed=" + std::to_string(network.num_link) +
            ", expected=" + std::to_string(expected_links)
        );
    }
}

void NetworkParser::read_trips(Network& network, const std::filesystem::path& file_path) const {
    auto lines = read_token_lines(file_path);

    double expected_total_flow = -1.0;
    std::size_t first_origin_line = lines.size();

    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto& line = lines[i];
        if (contains(line, "TOTAL")) {
            expected_total_flow = std::stod(line.back());
        }
        if (contains(line, "Origin")) {
            first_origin_line = i;
            break;
        }
    }

    if (first_origin_line == lines.size()) {
        throw std::runtime_error("Cannot parse trip table from " + file_path.string());
    }

    int origin_id = -1;
    double parsed_demand = 0.0;
    for (std::size_t i = first_origin_line; i < lines.size(); ++i) {
        const auto& line = lines[i];
        if (contains(line, "Origin")) {
            origin_id = std::stoi(line.back()) - 1;
            continue;
        }

        if (origin_id < 0) {
            continue;
        }

        for (std::size_t j = 0; j + 1 < line.size(); j += 2) {
            const int destination_id = std::stoi(line[j]) - 1;
            const double demand = std::stod(line[j + 1]) * network.demand_level;
            parsed_demand += demand;
            network.add_od(origin_id, destination_id, demand);
        }
    }

    if (expected_total_flow > 0.0) {
        const double expected = expected_total_flow * network.demand_level;
        if (std::abs(parsed_demand - expected) / expected > 0.01) {
            throw std::runtime_error(
                "Inconsistent demand in " + file_path.string() +
                ": parsed=" + std::to_string(parsed_demand) +
                ", expected=" + std::to_string(expected)
            );
        }
    }

    network.total_flow = 0.0;
    for (const auto& od : network.od_set) {
        network.total_flow += od.demand;
    }
}

}  // namespace ta
