// Created by andwh on 24/10/2024.

#include "simulation.cuh"
#include <cmath>

#define NUM_RUNS 100
#define NUM_COMPONENTS 2
#define MAX_COMPONENTS 100

// Et array af locations for et specifikt component
// En funktion der mapper en værdi i det array til den relevante node


// struct ComponentState {
//     struct RunState {
//         float current_time;
//         bool needs_sync;
//         float proposed_delay; // The sampled delay
//     };
//
//     __shared__ struct {
//         unordered_map<int, std::list<edge> > component_node_edge_map;
//         int component_start_node;
//         string template_name;
//     } component_data;
// };
//
// struct SimulationState {
//     ComponentState component_states[NUM_COMPONENTS][NUM_RUNS];
//
//     RunState run_states[NUM_RUNS];
// };


// __global__ void componentSimulation(SimulationState state) {
//     int component_id = blockIdx.x;
//     int run_id = threadIdx.x;
//
//     if (run_id >= NUM_COMPONENTS) return;
//
//     __shared__ ComponentInfo component_info;
//
//     if (threadIdx.x == 0) {
//         loadComponentInfo(&component_info, component_id);
//     }
//     __syncthreads(); // sync all the threads
//
//     while (!isSimulationDone(run_id)) {
//         // sim logic
//     }
//

//
//
//
//
// }

struct SyncRequest {
    int channel_id; // What channel to synchronize on
    bool is_sender; // ! or ?
    int component_id; // ID of the component that made the request
    bool is_urgent; // Is the channel urgent?
    bool is_broadcast; // Is the channel a broadcast channel?
};

// Shared memory synchronization buffer for a single run
struct SyncBuffer {
    SyncRequest requests[MAX_COMPONENTS]; // Array of synchronization requests from other components

    int number_of_requests; // Pending request number

    // The result of synchronization. Matched pairs that can synchronize.

    struct {
        int sender_id;
        int receiver_id;
        int channel_id;
        bool is_broadcast;
    } matches[MAX_MATCHES];
    int num_matches;

    // For broadcast channels
    int num_receivers; // How many received a broadcast

    // https://docs.uppaal.org/language-reference/system-description/templates/edges/
    // Search for broadcast channels



};

// We calculate max components by: max instances of all components added together

__global__ void simulateRun() {
    // Setting up the threads
    int thread_id = threadIdx.x;
    int component_id = thread_id / 32; // Since we have a specific component in each warp, we get the component type
    int warp_offset = thread_id % 32; // The position within the warp

    // Shared memory setup

    __shared__ float shared_delays[MAX_COMPONENTS];
    __shared__ float min_delay;
    __shared__
}

__global__ void sortArray(float *input, float *result, int nblocks) {
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = blockDim.x * nblocks;
    for (x = 0; i < log2(nThreads); i++) {
        if (thread) {

        }
    }



}


void simulation::runSimulation() {
    // Problem with models, spawning new components Trains in train gate for example?
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
