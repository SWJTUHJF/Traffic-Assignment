#pragma once

#include <cstddef>
#include <string>
#include <vector>

enum class CostType {
    Cost,
    MarginalCost,
};

struct Node {
    int node_id{};
    std::vector<int> link_in;
    std::vector<int> link_out;

    explicit Node(int id = 0) : node_id(id) {}
};

struct Link {
    int link_id{};
    int tail_id{};
    int head_id{};
    double capacity{};
    double length{};
    double free_flow_time{};
    double alpha{0.15};
    double beta{4.0};

    double flow{};
    double aux_flow{};
    double cost{};
    double marginal_cost{};

    Link() = default;

    Link(
        int link_id,
        int tail_id,
        int head_id,
        double capacity,
        double length,
        double free_flow_time,
        double alpha,
        double beta
    );

    [[nodiscard]] double get_cost(double x) const;
    [[nodiscard]] double get_marginal_cost(double x) const;
    [[nodiscard]] double weight(CostType cost_type) const;
    void update_cost();
    void update_marginal_cost();
    void update_cost_and_marginal_cost();
};

struct OD {
    int origin_id{};
    int destination_id{};
    double demand{};
};

struct Path {
    int origin_id{};
    int destination_id{};
    std::vector<int> link_ids;
    double cost{};

    [[nodiscard]] std::vector<int> node_ids(const std::vector<Link>& links) const;
};

class Network {
public:
    std::string name;
    double demand_level{1.0};

    std::vector<Node> node_set;
    std::vector<Link> link_set;
    std::vector<OD> od_set;

    int num_node{};
    int num_link{};
    int num_od{};
    double total_flow{};

    explicit Network(std::string name = "", double demand_level = 1.0);

    void resize_nodes(int count);
    Link& add_link(
        int tail_id,
        int head_id,
        double capacity,
        double length,
        double free_flow_time,
        double alpha,
        double beta
    );
    void add_od(int origin_id, int destination_id, double demand);

    void update_all_link_cost();
    void update_all_link_marginal_cost();
    void update_all_link_cost_and_marginal_cost();

    [[nodiscard]] double tstt() const;
    [[nodiscard]] double tsmtt() const;
};

