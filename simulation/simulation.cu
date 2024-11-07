// Created by andwh on 24/10/2024.

#include "simulation.cuh"
#include <cmath>

#include "state/SharedModelState.cuh"
#include "state/SharedRunState.cuh"

#define NUM_RUNS 6
#define NUM_COMPONENTS 2
#define MAX_COMPONENTS 100
#define TIME_BOUND 1.0

// Et array af locations for et specifikt component
// En funktion der mapper en værdi i det array til den relevante node



int get_total_runs(float confidence, float precision) {
    // confidence level = alpha, i.e. 0.05 for 95% confidence
    // precision = epsilon, i.e. 0.01 for +-1% error

    // int total_runs = (int)ceil(log(2.0/confidence)/log(2.0*precision*precision));
    // int total_runs = static_cast<int>(ceil(log(2.0 / confidence) / log(2.0 * precision * precision)));
    int total_runs = static_cast<size_t>(ceil((log(2.0) - log(confidence)) / (2*pow(precision, 2))));
    return total_runs;
}

__global__ void simulation_kernel(SharedModelState* model, bool* results, int runs_per_block, float time_bound, float confidence, float precision) {
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;


    // Pattern used to initialize shared memory for more complex data structures
    // __shared__ char shared_mem[]; // flexible, dynamically-sized array
    // SharedRunState* run_state = (SharedRunState*)shared_mem; // cast base address of shared_mem array to a pointer of type SharedRunState. Effectively maps it to type of SharedRunState
    // run_state now points to the start of the shared memory

    // for (int run = 0; run < runs_per_block; run++) {
    //     // Initialize the state for a run
    //     run_state->init(model);
    //
    //     // Main simulation loop for the current run
    //     while (run_state->global_time < time_bound) {
    //         // Each thread (component) computes its own delay
    //         run_state->compute_delays(model);
    //     }
    // }




}

void simulation::run_statistical_model_checking(SharedModelState* model, float confidence, float precision) {
    int total_runs = get_total_runs(confidence, precision);
    cout << "total_runs = " << total_runs << endl;

    // Kernel launch params
    int threads_per_block = 128;
    int runs_per_block = 1;
    int num_blocks = (total_runs + runs_per_block - 1) / runs_per_block; // + runs_per_block - 1 to round up (ensure enough blocks)
    // ---

    bool* device_results;
    cudaMalloc(&device_results, total_runs * sizeof(bool));

    // Main simulation kernel
    simulation_kernel<<<num_blocks, threads_per_block>>>(model, device_results, runs_per_block, TIME_BOUND, confidence, precision);


    // Collect results so we can later analyze them and compute statistics
    bool* host_results = new bool[total_runs];
    cudaMemcpy(host_results, device_results, total_runs * sizeof(bool),
               cudaMemcpyDeviceToHost);

    // Compute statistics
    //TODO: analyze_results(host_results, total_runs, confidence, precision);





}





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
