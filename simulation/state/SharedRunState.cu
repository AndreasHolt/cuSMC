//
// Created by andwh on 05/11/2024.
//

#include "SharedRunState.cuh"
#define START_NODE_ID 1 // TODO: get this from the model. But usually it's 1


// https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/
// Restrict tells the compiler that the pointer is the only way to access this data.
// Giving a pointer the restrict property, the programmer is promising the compiler that any data written
// to through that pointer is not read by any other pointer with the restrict property
// Not as important on Ampere as Kepler, since Ampere doesn't have dedicated readonly cache

// __device__ void SharedRunState::init(const SharedModelState* __restrict__ model) {
//     if(threadIdx.x < model->num_components) {
//         // Each thread initializes one component
//         components[threadIdx.x].current_node_id = START_NODE_ID;
//         components[threadIdx.x].next_delay = INFINITY;
//         components[threadIdx.x].next_edge_index = -1;
//         components[threadIdx.x].committed = false;
//         components[threadIdx.x].urgent = false;
//     }
//
//     if(threadIdx.x == 0) {
//         global_time = 0.0f;
//     }
//     __syncthreads();
// }


// __device__ void SharedRunState::compute_delays(const SharedModelState* model) {
//     int comp_id = threadIdx.x;
//     if (comp_id >= model->num_components) return;
//
//     ComponentState& comp = components[comp_id]; // we get the component state for the current thread. Just index by thread id (comp_id)
//
//     // Get the current information using the level it has stored
//     const NodeInfo& current_node = model->nodes[comp.current_node_id];
//     const NodeInfo& node = model->nodes[current_node.level * model->num_components + comp_id];
//
//     float min_delay = INFINITY;
//     float max_delay = INFINITY;

    // Check location invariant (if exists)
    // This sets upper bound for delay
    // for(int g = 0; g < node.num_guards; g++) {
    //     const GuardInfo& guard = model->guards[node.guards_start_index + g];
    //     if(is_invariant(guard)) {
    //         float bound = evaluate_guard_bound(guard, comp.clock_values);
    //         max_delay = min(max_delay, bound);
    //     }
    // }




// }