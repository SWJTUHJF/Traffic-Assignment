#pragma once

#include "traffic_assignment/network.hpp"

#include <filesystem>
#include <string>

namespace ta {

class NetworkParser {
public:
    explicit NetworkParser(std::filesystem::path data_root = "data");

    [[nodiscard]] Network load(const std::string& name, double demand_level = 1.0) const;

private:
    std::filesystem::path data_root_;

    [[nodiscard]] std::filesystem::path resolve_file(const std::string& name, const std::string& filename) const;
    void read_network(Network& network, const std::filesystem::path& file_path) const;
    void read_trips(Network& network, const std::filesystem::path& file_path) const;
};

}  // namespace ta
