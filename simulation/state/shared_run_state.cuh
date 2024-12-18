//
// Created by andwh on 05/11/2024.
//

#ifndef SHAREDRUNSTATE_CUH
#define SHAREDRUNSTATE_CUH
#include "shared_model_state.cuh"



// Minimal shared state needed per component
struct alignas(4) SharedComponentState {
    uint16_t component_id;
    bool has_delay;
    float next_delay;
};

// Thread-local state in registers
struct ThreadLocalState {
    uint8_t num_enabled_edges;
    uint16_t enabled_edges[MAX_EDGES_PER_NODE];
    const NodeInfo *current_node;
};


struct SharedBlockMemory {
    float global_time;
    unsigned simulation_id;
    bool has_hit_goal;
    double query_variable_min;
    double query_variable_max;

    // Variables (fixed size array in shared memory)
    struct Variable {
        float value;
        uint8_t rate;
        VariableKind kind;
    } variables[MAX_VARIABLES];

    int num_variables;

    // Synchronization
    int ready_count;
    bool has_urgent;
    bool has_committed;

    // Broadcast channels
    int channel_active;
    int channel_sender;

    // Static initialization method
    __device__ static void init(SharedBlockMemory *shared, int sim_id) {
        shared->global_time = 0.0;
        shared->simulation_id = sim_id;
        shared->ready_count = 0;
        shared->has_urgent = false;
        shared->has_committed = false;
        shared->has_hit_goal = false;
        shared->query_variable_min = 0.0;
        shared->query_variable_max = 0.0;

        // Initialize variables explicitly
        for (int i = 0; i < MAX_VARIABLES; i++) {
            shared->variables[i].value = 0.0;
            shared->variables[i].rate = 1;
            shared->variables[i].kind = VariableKind::INT;
        }

        // Clear channels
        shared->channel_active = false;
        shared->channel_sender = -1;
    }
};

struct BlockSimulationState {
    SharedComponentState *my_component; // Points to thread's component in shared memory
    curandState *random; // Thread's RNG state
};

#endif //SHAREDRUNSTATE_CUH
