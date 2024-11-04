//
// Created by andwh on 04/11/2024.
//
#include "../../engine/Domain.h"

#ifndef SHAREDMODELSTATE_CUH
#define SHAREDMODELSTATE_CUH




// struct SharedModelState {
//     const node** nodes;           // Array of automata nodes
//     const int num_automata;       // Number of automata components
//     const clock_var* variables;   // Array of clock/data variables
//     const int num_variables;       // Number of variables
//     const expr** expressions;     // Array of expressions used in model
//     const int num_expressions;     // Number of expressions
//     const edge** edges;           // Array of transitions
//     const int num_edges;          // Number of edges
//     const bool* initial_urgent;   // Initial urgent states
//     const bool* initial_committed; // Initial committed states
//     const unsigned max_expr_depth; // For expression evaluation
//     const unsigned max_backtrace_depth;
//     const unsigned max_edge_fanout; // Maximum outgoing edges
// };


struct NodeInfo {
    int id;
    int type;
    expr* lambda;

    CPU GPU NodeInfo() : id(0), type(0), lambda(nullptr) {}

    // Add constructor to allow initialization
    CPU GPU NodeInfo(int i, int t, expr* l) :
        id(i), type(t), lambda(l) {}
};

struct SharedModelState {
    // component-level information
    const int num_components;
    const int* component_sizes;
    const NodeInfo* nodes;  // [comp0_node0, comp1_node0, comp2_node0, comp0_node1, comp1_node1, ...]

    // constructor to allow const members (attempt to force L1 cache)
    CPU GPU SharedModelState(int nc, const int* cs, const NodeInfo* n) :
        num_components(nc),
        component_sizes(cs),
        nodes(n) {}
};


SharedModelState* init_shared_model_state(
    const network* cpu_network,
    const std::unordered_map<int, int>& node_subsystems_map,
    const std::unordered_map<int, node*>& node_map);


__global__ void test_kernel(SharedModelState* model);


#endif //SHAREDMODELSTATE_CUH
