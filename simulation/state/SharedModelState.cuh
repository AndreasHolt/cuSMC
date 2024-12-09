//
// Created by andwh on 04/11/2024.
//



#ifndef SHAREDMODELSTATE_CUH
#define SHAREDMODELSTATE_CUH

#include "../../include/VariableTypes.h"
#include "../../engine/Domain.h"
#include "../../automata_parser/VariableUsageVisitor.h"
#include <list>
#include <cuda_runtime.h>






class abstract_parser;
class uppaal_xml_parser;

#define MAX_VAR_NAME_LENGTH 128

struct VariableInfo {
    int variable_id;
    VariableKind type;
    char name[MAX_VAR_NAME_LENGTH];
    double initial_value;

    CPU GPU VariableInfo(int id, VariableKind t, const char* n, double val = 0.0)
        : variable_id(id), type(t), initial_value(val) {
        strncpy(name, n, MAX_VAR_NAME_LENGTH - 1);
        name[MAX_VAR_NAME_LENGTH - 1] = '\0';
    }

    CPU GPU VariableInfo()
        : variable_id(0), type(VariableKind::INT), initial_value(0.0) {
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
    VariableKind kind;

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

    // Old constructor for backward compatibility (in case it breaks something)
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


// Helper for safely accessing expr pointer
__device__ __forceinline__ const expr* fetch_expr(const expr* ptr) {
    // return ptr since we can't __ldg a whole struct directly.
    return ptr;
}

__device__ __forceinline__ double fetch_expr_value(const expr* ptr) {
#if __CUDA_ARCH__ >= 350
    // printf("Using __ldg for fetch_expr_value, ptr->value: %f\n", ptr ? ptr->value : 0.0);
    return ptr ? __ldg((const double*)&ptr->value) : 0.0;
#else
    // printf("Using regular load for fetch_expr_value, ptr->value: %f\n", ptr ? ptr->value : 0.0);
    return ptr ? ptr->value : 0.0;
#endif
}

__device__ __forceinline__ int fetch_expr_operand(const expr* ptr) {
#if __CUDA_ARCH__ >= 350
    return ptr ? __ldg((const int*)&ptr->operand) : -1;
#else
    return ptr ? ptr->operand : -1;
#endif
}


/*
 Caching techniques:
 * We use __restrict__ for main data arrays, to enable L1 caching. We know that these arrays are laid out nicely and accessed predictably.
*/

struct SharedModelState {
    const int num_components;
    const int max_nodes_per_component;
    const int* const __restrict__ component_sizes;
    const NodeInfo* const __restrict__ nodes;
    const EdgeInfo* const __restrict__ edges;
    const GuardInfo* const __restrict__ guards;
    const UpdateInfo* const __restrict__ updates;
    const GuardInfo* const __restrict__ invariants;
    const int* const __restrict__ initial_var_values;

    // Default constructor
    CPU GPU SharedModelState() :
        num_components(0),
        max_nodes_per_component(0),
        component_sizes(nullptr),
        nodes(nullptr),
        edges(nullptr),
        guards(nullptr),
        updates(nullptr),
        invariants(nullptr),
        initial_var_values(nullptr) {}

    CPU GPU SharedModelState(
        const int nc, const int mnpc, const int* cs,
        const NodeInfo* n, const EdgeInfo* e,
        const GuardInfo* g, const UpdateInfo* u, const GuardInfo* i, const int* iv) :
            num_components(nc),
            max_nodes_per_component(mnpc),
            component_sizes(cs),
            nodes(n),
            edges(e),
            guards(g),
            updates(u),
            invariants(i),
            initial_var_values(iv) {}
};

SharedModelState* init_shared_model_state(
    const network* cpu_network,
    const std::unordered_map<int, int>& node_subsystems_map,
    const std::unordered_map<int, std::list<edge>>& node_edge_map,
    const std::unordered_map<int, node*>& node_map,
    const std::unordered_map<int, VariableTrackingVisitor::VariableUsage>& variable_registry,
    const abstract_parser* parser, const int num_vars);


__global__ void test_kernel(SharedModelState* model);
__global__ void validate_edge_indices(SharedModelState* model);
__global__ void verify_expressions_kernel(SharedModelState* model);



#endif //SHAREDMODELSTATE_CUH
