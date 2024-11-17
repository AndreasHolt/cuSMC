//
// Created by andwh on 05/11/2024.
//

#ifndef SHAREDRUNSTATE_CUH
#define SHAREDRUNSTATE_CUH
#include "SharedModelState.cuh"
#define MAX_COMPONENTS 2

//TODO: Make the below dynamic based on analysis of the model
constexpr int MAX_VALUE_STACK_SIZE = 64;  // Can handle deeply nested expressions
constexpr int MAX_CHANNELS = 32;          // Can handle many channels
constexpr int MAX_VARIABLES = 32;         // Can handle many variables
#define MAX_EDGES_PER_NODE 10


namespace Constants {
    constexpr int MAX_VARIABLES = 32;

}

//class SharedRunState {
//
//};

struct ComponentState {
    int component_id;
    const NodeInfo* current_node;
    double next_delay;
    bool has_delay;
    int enabled_edges[MAX_EDGES_PER_NODE];  // Store indices of enabled edges TODO: replace with max fanout that we get with optimizer
    int num_enabled_edges;                  // Number of currently enabled edges
};

// This replaces both SharedSimState and SharedBlockMemory
// Contains actual arrays instead of pointers since it will be in shared memory
struct SharedBlockMemory {
    // Global simulation state
    double global_time;
    unsigned urgent_count;
    unsigned committed_count;
    unsigned simulation_id;

    // Variables (fixed size array in shared memory)
    struct Variable {
        double value;
        int rate;
        VariableKind kind;
        int last_writer;
    } variables[MAX_VARIABLES];
    int num_variables;

    // Expression evaluation stack
    double value_stack[MAX_VALUE_STACK_SIZE];
    int value_stack_top;

    // Synchronization
    int ready_count;
    bool has_urgent;
    bool has_committed;

    // Broadcast channels
    bool channel_active[MAX_CHANNELS];
    int channel_sender[MAX_CHANNELS];

    // Static initialization method
    __device__ static void init(SharedBlockMemory* shared, int sim_id) {
        shared->global_time = 0.0;
        shared->urgent_count = 0;
        shared->committed_count = 0;
        shared->simulation_id = sim_id;
        shared->value_stack_top = 0;
        shared->ready_count = 0;
        shared->has_urgent = false;
        shared->has_committed = false;

        // Initialize variables explicitly
        for(int i = 0; i < Constants::MAX_VARIABLES; i++) {
            shared->variables[i].value = 0.0;
            shared->variables[i].rate = 1;
            shared->variables[i].kind = VariableKind::INT;
            shared->variables[i].last_writer = -1;
        }

        // Clear channels
        for(int i = 0; i < MAX_CHANNELS; i++) {
            shared->channel_active[i] = false;
            shared->channel_sender[i] = -1;
        }
    }

};

// This becomes simpler as it just points to everything needed
struct BlockSimulationState {
    SharedModelState* model;          // Global memory
    SharedBlockMemory* shared;        // Shared memory
    ComponentState* my_component;      // Points to thread's component in shared memory
    curandState* random;              // Thread's RNG state
};



// ----------------------------------------------------------

// struct ComponentStateOld {
//     int current_node_id;     // Current node the component is in
//     float* clock_values;     // Array of clock values for this component
//     float next_delay;        // Pre-computed next delay
//     int next_edge_index;     // Pre-computed next edge to take (-1 if none)
//     bool committed;          // If in committed location
//     bool urgent;             // If in urgent location
// };
//
// struct SharedRunState {
//     // These are in shared memory, one instance per block
//     ComponentStateOld components[MAX_COMPONENTS];
//     float global_time;       // Current simulation time
//     float* shared_clocks;    // Shared clock values
//
//     // Methods for state management
//     __device__ void init(const SharedModelState* model);
//     __device__ void compute_delays(const SharedModelState* model);
//     __device__ void advance_time(float time_delta);
//     __device__ void take_transition(const SharedModelState* model, int component_id, int edge_index);
// };
//





#endif //SHAREDRUNSTATE_CUH
