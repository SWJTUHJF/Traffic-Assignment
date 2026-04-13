#include <iostream>
#include <print>
#include <string>
#include <vector>


class Link;


class Node{
    public:
    const int node_id;
    std::vector<Link*> link_in;
    std::vector<Link*> link_out;

    Node(int node_id): node_id(node_id) {};

    void add_in_link(Link* link){
        link_in.push_back(link);
    }

    void add_out_link(Link* link){
        link_out.push_back(link);
    }
};


class Link{
    private:
        const double capacity{0.0};
        const double length{0.0};
        const double free_flow_time{0.0};
        const double alpha{0.15};
        const double beta{4.0};
        double flow{0.0};
        double cost{0.0};
        double marginal_cost{0.0};

    public:
        const int link_id;
        const Node* tail_node;
        const Node* head_node;

    Link(
        int link_id,
        const Node* tail_node,
        const Node* head_node,
        double capacity = 0.0,
        double length = 0.0,
        double free_flow_time = 0.0,
        double alpha = 0.15,
        double beta = 4.0
    ): 
        capacity(capacity),
        length(length),
        free_flow_time(free_flow_time),
        alpha(alpha),
        beta(beta),
        link_id(link_id),
        tail_node(tail_node),
        head_node(head_node) {}
};


int main(){
    Node node_1 = Node(1);
    std::print("{}",node_1.node_id);
};