//
// Created by andwh on 05/11/2024.
//

#ifndef SHAREDRUNSTATE_CUH
#define SHAREDRUNSTATE_CUH
#include "shared_model_state.cuh"



struct alignas(8) ComponentState {
    // We force 4-byte alignment instead of 8
    // Pack these 4 bytes
    uint16_t component_id;
    uint8_t num_enabled_edges;
    bool has_delay;

    // Pack 4 bytes, naturally aligned
    float next_delay;

    uint16_t enabled_edges[MAX_EDGES_PER_NODE];

    const NodeInfo *current_node; // 8 bytes, put pointer last
};

// Variables (fixed size array in shared memory)
struct Variable {
    float value;
    int rate;
    VariableKind kind;
    int last_writer;
};

struct SharedBlockMemory {
    float global_time;
    unsigned simulation_id;
    bool has_hit_goal;
    double query_variable_min;
    double query_variable_max;

    Variable *variables;

    int num_variables;

    // Synchronization
    int ready_count;
    bool has_urgent;
    bool has_committed;

    // Broadcast channels
    bool channel_active[MAX_CHANNELS];
    int channel_sender[MAX_CHANNELS];

    // Static initialization method
    __device__ static void init(SharedBlockMemory *shared, int sim_id, Variable* vars, int num_vars) {
        shared->global_time = 0.0;
        shared->simulation_id = sim_id;
        shared->ready_count = 0;
        shared->has_urgent = false;
        shared->has_committed = false;
        shared->has_hit_goal = false;
        shared->query_variable_min = 0.0;
        shared->query_variable_max = 0.0;
        shared->variables = vars;

        shared->num_variables = num_vars;
        // Initialize variables explicitly
        for (int i = 0; i < num_vars; i++) {
            shared->variables[i] = {0.0, 1, VariableKind::INT, -1};
        }

        // Clear channels
        for (int i = 0; i < MAX_CHANNELS; i++) {
            shared->channel_active[i] = false;
            shared->channel_sender[i] = -1;
        }
    }
};

struct BlockSimulationState {
    ComponentState *my_component; // Points to thread's component in shared memory
    curandState *random; // Thread's RNG state
};

#endif //SHAREDRUNSTATE_CUH
