//
// Created by andwh on 05/11/2024.
//

#ifndef SHAREDRUNSTATE_CUH
#define SHAREDRUNSTATE_CUH
#include "SharedModelState.cuh"
#define MAX_COMPONENTS 2


//class SharedRunState {
//
//};

struct ComponentState {
    int current_node_id;     // Current node the component is in
    float* clock_values;     // Array of clock values for this component
    float next_delay;        // Pre-computed next delay
    int next_edge_index;     // Pre-computed next edge to take (-1 if none)
    bool committed;          // If in committed location
    bool urgent;             // If in urgent location
};

struct SharedRunState {
    // These are in shared memory, one instance per block
    ComponentState components[MAX_COMPONENTS];
    float global_time;       // Current simulation time
    float* shared_clocks;    // Shared clock values

    // Methods for state management
    __device__ void init(const SharedModelState* model);
    __device__ void compute_delays(const SharedModelState* model);
    __device__ void advance_time(float time_delta);
    __device__ void take_transition(const SharedModelState* model, int component_id, int edge_index);
};




#endif //SHAREDRUNSTATE_CUH
