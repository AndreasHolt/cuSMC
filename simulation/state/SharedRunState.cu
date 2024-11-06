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

__device__ void SharedRunState::init(const SharedModelState* __restrict__ model) {
    if(threadIdx.x < model->num_components) {
        // Each thread initializes one component
        components[threadIdx.x].current_node_id = START_NODE_ID;
        components[threadIdx.x].next_delay = INFINITY;
        components[threadIdx.x].next_edge_index = -1;
        components[threadIdx.x].committed = false;
        components[threadIdx.x].urgent = false;
    }

    if(threadIdx.x == 0) {
        global_time = 0.0f;
    }
    __syncthreads();
}


__device__ void SharedRunState::compute_delays(const SharedModelState* model) {
    int comp_id = threadIdx.x;
    if (comp_id >= model->num_components) return;

    ComponentState& comp = components[comp_id]; // we get the component state for the current thread. Just index by thread id (comp_id)

    // Get the current information using coalesced memory access




}