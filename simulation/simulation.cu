//
// Created by andwh on 24/10/2024.
//

#include "simulation.cuh"

#define NUM_RUNS 100
#define NUM_COMPONENTS 100

// Et array af locations for et specifikt component
// En funktion der mapper en værdi i det array til den relevante node

// struct ComponentInfo {
//
// };


__global__ void componentSimulation(SimulationState state) {
    int component_id = blockIdx.x;
    int run_id = threadIdx.x;

    if (run_id >= NUM_COMPONENTS) return;

    __shared__ ComponentInfo component_info;

    if (threadIdx.x == 0) {
        loadComponentInfo(&component_info, component_id);
    }
    __syncthreads(); // sync all the threads

    while (!isSimulationDone(run_id)) {
        // sim logic
    }





}

void simulation::runSimulation() {
    componentSimulation<<<NUM_COMPONENTS, 128>>>(); // 2 blocks (component size), 100 simulations but round up to 128


    // Pick delays: implement delay function
    // Find the smallest delay, and which index it has (to find component it belongs to)
    // Apply the delay
    // Pick a transition from the component that won: Pick according to the weights
    // Check whether we need to synchronize with anything when taking this transition
    // Take the transition

    // We need the state such that we can describe the run afterwards. We add our delays to it.
    cout << "test from run sim" << endl;
}


