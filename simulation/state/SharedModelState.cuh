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

struct EdgeInfo {
    int source_node_id;       // ID of source node
    int dest_node_id;         // ID of destination node
    int channel;             // Channel for synchronization
    expr* weight;            // Weight expression

    // Guards and updates info
    int num_guards;          // Number of guards for this edge
    int guards_start_index;  // Index into guards array where this edge's guards start
    int num_updates;         // Number of updates
    int updates_start_index; // Index into updates array

    // Constructor
    CPU GPU EdgeInfo() : source_node_id(0), dest_node_id(0), channel(0),
                        weight(nullptr), num_guards(0), guards_start_index(0),
                        num_updates(0), updates_start_index(0) {}

    CPU GPU EdgeInfo(int src, int dst, int ch, expr* w,
                    int ng, int gs, int nu, int us) :
        source_node_id(src), dest_node_id(dst), channel(ch), weight(w),
        num_guards(ng), guards_start_index(gs),
        num_updates(nu), updates_start_index(us) {}
};

struct GuardInfo {
    constraint::operators operand;
    bool uses_variable;
    union {
        expr* value;
        int variable_id;
        int compile_id;
    };
    expr* expression;

    // Default constructor
    CPU GPU GuardInfo() : operand(constraint::less_equal_c),
                         uses_variable(false), value(nullptr),
                         expression(nullptr) {}

    // Constructor for initialization with all fields
    CPU GPU GuardInfo(constraint::operators op, bool uv, expr* v, expr* exp) :
        operand(op),
        uses_variable(uv),
        expression(exp)
    {
        value = v;  // Sets the union's value field
    }
};

struct UpdateInfo {
    int variable_id;
    expr* expression;

    // Default constructor
    CPU GPU UpdateInfo() : variable_id(0), expression(nullptr) {}

    // Constructor for initialization with all fields
    CPU GPU UpdateInfo(int vid, expr* exp) :
        variable_id(vid),
        expression(exp) {}
};


struct NodeInfo {
    int id;
    int type;
    int level;
    expr* lambda;
    int first_edge_index;
    int num_edges;
    int first_invariant_index;
    int num_invariants;

    CPU GPU NodeInfo() :
        id(0), type(0), level(0), lambda(nullptr),
        first_edge_index(-1), num_edges(0), first_invariant_index(-1), num_invariants(0) {}

    CPU GPU NodeInfo(int i, int t, int l, expr* lam, int fei, int ne, int fii, int ni) :
        id(i), type(t), level(l), lambda(lam),
        first_edge_index(fei), num_edges(ne), first_invariant_index(fii), num_invariants(ni) {}
};



struct SharedModelState {
    const int num_components;
    const int* component_sizes;
    const NodeInfo* nodes;
    const EdgeInfo* edges;
    const GuardInfo* guards;
    const UpdateInfo* updates;
    const GuardInfo* invariants;

    CPU GPU SharedModelState(
        int nc, const int* cs,
        const NodeInfo* n, const EdgeInfo* e,
        const GuardInfo* g, const UpdateInfo* u, const GuardInfo* i) :
            num_components(nc),
            component_sizes(cs),
            nodes(n),
            edges(e),
            guards(g),
            updates(u),
            invariants(i) {}
};


//
// SharedModelState* init_shared_model_state(
//     const network* cpu_network,
//     const std::unordered_map<int, int>& node_subsystems_map,
//     const std::unordered_map<int, node*>& node_map);

SharedModelState* init_shared_model_state(
    const network* cpu_network,
    const std::unordered_map<int, int>& node_subsystems_map,
    const std::unordered_map<int, std::list<edge>>& node_edge_map, const std::unordered_map<int, node*>& node_map);


__global__ void test_kernel(SharedModelState* model);
__global__ void validate_edge_indices(SharedModelState* model);
__global__ void verify_invariants_kernel(SharedModelState* model);

#endif //SHAREDMODELSTATE_CUH
