//
// Created by andwh on 04/11/2024.
//



#ifndef SHAREDMODELSTATE_CUH
#define SHAREDMODELSTATE_CUH

#include "../../include/VariableTypes.h"
#include "../../engine/Domain.h"
#include "../../automata_parser/VariableUsageVisitor.h"


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

class abstract_parser;
class uppaal_xml_parser;



#define MAX_VAR_NAME_LENGTH 128



struct VariableInfo {
    int variable_id;
    VariableKind type;
    char name[MAX_VAR_NAME_LENGTH];

    CPU GPU VariableInfo(int id, VariableKind t, const char* n)
        : variable_id(id), type(t) {
        strncpy(name, n, MAX_VAR_NAME_LENGTH - 1);
        name[MAX_VAR_NAME_LENGTH - 1] = '\0';
    }

    CPU GPU VariableInfo()
        : variable_id(0), type(VariableKind::INT) {
        name[0] = '\0';
    }
};


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
        VariableInfo var_info;
        struct {
            expr* value;
            expr* expression;
        };
    };

    // Constructor for variable-based guard
    CPU GPU GuardInfo(constraint::operators op, const VariableInfo& var, expr* expr)
    : operand(op), uses_variable(true) {
        var_info = var;
        expression = expr;  // Need to set this after var_info since they share a union
    }



    // Constructor for value-based guard
    CPU GPU GuardInfo(constraint::operators op, bool uses_var, expr* val, expr* expr)
        : operand(op), uses_variable(uses_var), value(val), expression(expr) {}


    // Default constructor needed for vector
    CPU GPU GuardInfo() : operand(constraint::less_equal_c), uses_variable(false), value(nullptr), expression(nullptr) {}
};

struct UpdateInfo {
    int variable_id;
    expr* expression;
    VariableKind kind;  // Add variable kind

    // Default constructor
    CPU GPU UpdateInfo() :
        variable_id(0),
        expression(nullptr),
        kind(VariableKind::INT) {}  // Default to INT_LOCAL

    // Constructor for initialization with all fields
    CPU GPU UpdateInfo(int vid, expr* exp, VariableKind k) :
        variable_id(vid),
        expression(exp),
        kind(k) {}

    // Keep old constructor for backward compatibility if needed
    CPU GPU UpdateInfo(int vid, expr* exp) :
        variable_id(vid),
        expression(exp),
        kind(VariableKind::INT) {}  // Default to INT_LOCAL
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

    // Default constructor
    CPU GPU SharedModelState() :
        num_components(0),
        component_sizes(nullptr),
        nodes(nullptr),
        edges(nullptr),
        guards(nullptr),
        updates(nullptr),
        invariants(nullptr) {}

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
    const std::unordered_map<int, std::list<edge>>& node_edge_map,
    const std::unordered_map<int, node*>& node_map,
    const std::unordered_map<int, VariableTrackingVisitor::VariableUsage>& variable_registry,
    const abstract_parser* parser);




__global__ void test_kernel(SharedModelState* model);
__global__ void validate_edge_indices(SharedModelState* model);
// __global__ void verify_invariants_kernel(SharedModelState* model);
__global__ void verify_expressions_kernel(SharedModelState* model);



#endif //SHAREDMODELSTATE_CUH
