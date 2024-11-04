// Created by andwh on 24/10/2024.

#include "simulation.cuh"
#include <cmath>

#define NUM_RUNS 6
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




__global__ void findSmallestElementInArray(float *input, int input_length, float *result, int nblocks) {
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = blockDim.x * nblocks;
    for (int i = 0; i < ceil(log2f(nThreads)); i++) { //1
        if (threadid % static_cast<int>(pow(2, i+1)==0)) {
            int correspondant = min(static_cast<int>(threadid + pow(2, i)), input_length-1);
            input[threadid] = min(input[threadid], input[correspondant]);
        }
    }
    if (threadid == 0) {
        *result = input[0];
    }

}
void testFunction () {
    float* h_a = new float[NUM_RUNS];
    srand( static_cast<unsigned>(time(NULL)));
    int upper = 3500;
    int lower = 1230;
    for (int i = 0; i < NUM_RUNS; i++) {

        h_a[i] = rand() % (upper - lower) + lower;
        cout << h_a[i] << ", ";
    }
    cout << endl;

    float* d_a;
    float d_result;
    cudaMalloc(&d_a, NUM_RUNS * sizeof(float));

    cudaMemcpy(d_a, h_a, NUM_RUNS * sizeof(float), cudaMemcpyHostToDevice);

    findSmallestElementInArray<<<1, 128>>>(d_a, NUM_RUNS, &d_result, 1); // 2 blocks (component size), 100 simulations but round up to 128

    cudaMemcpy(h_a, d_a, NUM_RUNS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_RUNS; i++) {
        cout << h_a[i] << ", ";
    }
    cout <<endl << "Result = " << d_result << endl;
}

void simulation::runSimulation() {
    // Problem with models, spawning new components Trains in train gate for example?
    // componentSimulation<<<NUM_COMPONENTS, 128>>>(); // 2 blocks (component size), 100 simulations but round up to 128

    // testFunction();

    // Pick delays: implement delay function
    // Find the smallest delay, and which index it has (to find component it belongs to)
    // Apply the delay
    // Pick a transition from the component that won: Pick according to the weights
    // Check whether we need to synchronize with anything when taking this transition
    // Take the transition

    // We need the state such that we can describe the run afterwards. We add our delays to it.



    cout << "test from run sim" << endl;
}
