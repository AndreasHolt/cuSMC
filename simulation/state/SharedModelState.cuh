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


struct SharedModelState {
    const node** nodes;           // All nodes
    const int total_nodes;        // Total number of nodes
    const int num_components;     // Number of components
    const int* component_starts;  // Index where each component's nodes start
    const int* component_sizes;   // Number of nodes in each component

    // Constructor
    CPU GPU SharedModelState(const node** n, int tn, int nc,
                           const int* cs, const int* csz) :
        nodes(n), total_nodes(tn), num_components(nc),
        component_starts(cs), component_sizes(csz) {}
};

SharedModelState* init_shared_model_state(
    const network* cpu_network,
    const std::unordered_map<int, int>& node_subsystems_map,
    const std::unordered_map<int, node*>& node_map);


__global__ void test_kernel(SharedModelState* model);


#endif //SHAREDMODELSTATE_CUH
