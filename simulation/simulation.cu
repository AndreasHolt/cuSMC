// Created by andwh on 24/10/2024.

#include "simulation.cuh"
#include <cmath>

#include "state/SharedModelState.cuh"
#include "state/SharedRunState.cuh"

#define NUM_RUNS 6
// #define NUM_COMPONENTS 2
// #define MAX_COMPONENTS 100
#define TIME_BOUND 1.0

#define MAX_VARIABLES 5


// Et array af locations for et specifikt component
// En funktion der mapper en værdi i det array til den relevante node

__device__ void check_cuda_error(const char* location) {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error at %s: %s\n", location, cudaGetErrorString(error));
    }
}

#define CHECK_ERROR(loc) check_cuda_error(loc)


__device__ void compute_possible_delay(
    ComponentState* my_state,
    SharedBlockMemory* shared,
    SharedModelState* model,
    BlockSimulationState* block_state)  // Add this
{
    // Add debug prints
    printf("Thread %d: Computing delay for node %d\n",
           threadIdx.x, my_state->current_node->id);

    my_state->has_delay = false;

    const NodeInfo& node = *my_state->current_node;

    // Debug print invariants
    printf("Thread %d: Node has %d invariants starting at index %d\n",
           threadIdx.x, node.num_invariants, node.first_invariant_index);

    // Rest same as before but with more debug prints
    if (node.type == node::urgent || node.type == node::committed) {
        printf("Thread %d: Urgent/committed node - delay=0\n", threadIdx.x);
        my_state->next_delay = 0.0;
        my_state->has_delay = true;
        return;
    }

    double min_delay = 0.0;
    double max_delay = DBL_MAX;
    bool is_bounded = false;

    for(int i = 0; i < node.num_invariants; i++) {
        const GuardInfo& inv = model->invariants[node.first_invariant_index + i];
        if(inv.uses_variable) {
            const auto& var = shared->variables[inv.var_info.variable_id];
            printf("Thread %d: Checking invariant %d for variable %d (value=%f, rate=%d)\n",
                   threadIdx.x, i, inv.var_info.variable_id, var.value, var.rate);

            double curr_val = var.value;
            double rate = var.rate;

            if(rate == 0) {
                printf("Thread %d: Skipping invariant %d - rate is 0\n",
                       threadIdx.x, i);
                continue;
            }

            if(inv.operand == constraint::less_equal_c ||
               inv.operand == constraint::less_c) {
                double bound = 5.0; // TODO: evaluate expression
                double time_to_bound = (bound - curr_val) / rate;
                if(inv.operand == constraint::less_c) {
                    time_to_bound -= 1e-6;
                }
                printf("Thread %d: Invariant %d gives bound %f\n",
                       threadIdx.x, i, time_to_bound);

                max_delay = min(max_delay, time_to_bound);
                is_bounded = true;
            }
        }
    }

    if(is_bounded && min_delay < max_delay) {
        curandState* rng = block_state->random;
        double rand = curand_uniform(rng);
        my_state->next_delay = min_delay + (max_delay - min_delay) * rand;
        my_state->has_delay = true;
        printf("Thread %d: Sampled delay %f (min=%f, max=%f, rand=%f)\n",
               threadIdx.x, my_state->next_delay, min_delay, max_delay, rand);
    } else {
        printf("Thread %d: No delay computed (bounded=%d, min=%f, max=%f)\n",
               threadIdx.x, is_bounded, min_delay, max_delay);
    }
}




__device__ double find_minimum_delay(
    ComponentState* my_state,
    SharedBlockMemory* shared,
    const int num_components)
{
    // Each thread stores its delay in shared memory array
    __shared__ double delays[MAX_COMPONENTS];
    __shared__ int component_indices[MAX_COMPONENTS];

    // Store my delay if I have one
    if(my_state->has_delay) {
        delays[threadIdx.x] = my_state->next_delay;
        component_indices[threadIdx.x] = my_state->component_id;
    } else {
        delays[threadIdx.x] = DBL_MAX;
        component_indices[threadIdx.x] = -1;
    }
    __syncthreads();

    // Parallel reduction to find minimum
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            if(delays[threadIdx.x + stride] < delays[threadIdx.x]) {
                delays[threadIdx.x] = delays[threadIdx.x + stride];
                component_indices[threadIdx.x] = component_indices[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Result is in delays[0] and winning component in component_indices[0]
    return delays[0];
}


int get_total_runs(float confidence, float precision) {
    // confidence level = alpha, i.e. 0.05 for 95% confidence
    // precision = epsilon, i.e. 0.01 for +-1% error

    // int total_runs = (int)ceil(log(2.0/confidence)/log(2.0*precision*precision));
    // int total_runs = static_cast<int>(ceil(log(2.0 / confidence) / log(2.0 * precision * precision)));
    int total_runs = static_cast<size_t>(ceil((log(2.0) - log(confidence)) / (2*pow(precision, 2))));
    return total_runs;
}

// TODO: what if we want to spawn 50 trains? How do we do that?


__global__ void simulation_kernel(SharedModelState* model, bool* results,
                                int runs_per_block, float time_bound) {
    printf("Starting kernel: block=%d, thread=%d\n",
           blockIdx.x, threadIdx.x);
    CHECK_ERROR("kernel start");

    // Verify model pointer
    if(model == nullptr) {
        printf("Thread %d: NULL model pointer!\n", threadIdx.x);
        return;
    }

    __shared__ SharedBlockMemory shared_mem;
    __shared__ ComponentState components[MAX_COMPONENTS];
    __shared__ curandState rng_states[MAX_COMPONENTS];

    CHECK_ERROR("after shared memory declaration");

    // Debug model access
    printf("Thread %d: Attempting to access model, num_components=%d\n",
           threadIdx.x, model->num_components);
    CHECK_ERROR("after model access");

    // Setup block state
    BlockSimulationState block_state;
    block_state.model = model;
    block_state.shared = &shared_mem;
    block_state.my_component = &components[threadIdx.x];

    printf("Thread %d: Block state setup complete\n", threadIdx.x);
    CHECK_ERROR("after block state setup");

    // Initialize RNG
    int sim_id = blockIdx.x * runs_per_block;
    int comp_id = threadIdx.x;
    curand_init(1234 + sim_id * blockDim.x + comp_id, 0, 0,
                &rng_states[threadIdx.x]);
    block_state.random = &rng_states[threadIdx.x];

    printf("Thread %d: RNG initialized\n", threadIdx.x);
    CHECK_ERROR("after RNG init");

    // Initialize shared state
    if (threadIdx.x == 0) {
        printf("Block %d: Initializing shared memory\n", blockIdx.x);
        SharedBlockMemory::init(&shared_mem, sim_id);
    }
    __syncthreads();
    CHECK_ERROR("after shared memory init");




    // Initialize component state
    if(threadIdx.x >= model->num_components) {
        printf("Thread %d: Exiting - thread ID exceeds number of components\n",
               threadIdx.x);
        return;
    }

    ComponentState* my_state = block_state.my_component;
    my_state->component_id = comp_id;
    my_state->current_node = &model->nodes[comp_id];
    my_state->has_delay = false;

    printf("Thread %d: Component initialized, node_id=%d\n",
           threadIdx.x, my_state->current_node->id);
    CHECK_ERROR("after component init");

    // Main simulation loop
    while(shared_mem.global_time < time_bound) {
        printf("Thread %d: Time=%f\n", threadIdx.x, shared_mem.global_time);

        compute_possible_delay(my_state, &shared_mem, model, &block_state);
        CHECK_ERROR("after compute delay");
        __syncthreads();

        double min_delay = find_minimum_delay(my_state, &shared_mem, blockDim.x);
        CHECK_ERROR("after find minimum");
        printf("Thread %d: Minimum delay = %f\n", threadIdx.x, min_delay);

        if(threadIdx.x == 0) {
            shared_mem.global_time += min_delay;
            printf("Block %d: Advanced time to %f\n",
                   blockIdx.x, shared_mem.global_time);
        }
        __syncthreads();
    }

    printf("Thread %d: Simulation complete\n", threadIdx.x);
}




void simulation::run_statistical_model_checking(SharedModelState* model, float confidence, float precision) {
    int total_runs = 1;
    cout << "total_runs = " << total_runs << endl;

    // Detailed model validation
    if(model == nullptr) {
        cout << "Error: NULL model pointer" << endl;
        return;
    }

    // Print model pointer address
    cout << "Model pointer address: " << model << endl;

    // Try to access model components safely
    cudaError_t error;
    SharedModelState host_model;
    error = cudaMemcpy(&host_model, model, sizeof(SharedModelState), cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
        cout << "CUDA error copying model: " << cudaGetErrorString(error) << endl;
        return;
    }

    cout << "Model components: " << host_model.num_components << endl;

    // Print more model details
    cout << "Component sizes array at: " << host_model.component_sizes << endl;
    cout << "Nodes array at: " << host_model.nodes << endl;
    cout << "Edges array at: " << host_model.edges << endl;

    bool* device_results;
    error = cudaMalloc(&device_results, total_runs * sizeof(bool));
    if(error != cudaSuccess) {
        cout << "CUDA malloc error: " << cudaGetErrorString(error) << endl;
        return;
    }

    // Launch configuration
    int threads_per_block = 8;
    int runs_per_block = 1;
    int num_blocks = 1;

    cout << "Launching kernel with configuration:" << endl;
    cout << "  Blocks: " << num_blocks << endl;
    cout << "  Threads per block: " << threads_per_block << endl;
    cout << "  Time bound: " << TIME_BOUND << endl;

    // Launch kernel
    simulation_kernel<<<num_blocks, threads_per_block>>>(
        model, device_results, runs_per_block, TIME_BOUND);

    error = cudaGetLastError();
    if(error != cudaSuccess) {
        cout << "Kernel launch error: " << cudaGetErrorString(error) << endl;
        return;
    }

    cout << "Kernel launched successfully, waiting for completion..." << endl;

    error = cudaDeviceSynchronize();
    if(error != cudaSuccess) {
        cout << "Kernel execution error: " << cudaGetErrorString(error) << endl;
        return;
    }

    cout << "Kernel execution complete" << endl;
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

// void testFunction () {
//     float* h_a = new float[NUM_RUNS];
//     srand( static_cast<unsigned>(time(NULL)));
//     int upper = 3500;
//     int lower = 1230;
//     for (int i = 0; i < NUM_RUNS; i++) {
//
//         h_a[i] = rand() % (upper - lower) + lower;
//         cout << h_a[i] << ", ";
//     }
//     cout << endl;
//
//     float* d_a;
//     float d_result;
//     cudaMalloc(&d_a, NUM_RUNS * sizeof(float));
//
//     cudaMemcpy(d_a, h_a, NUM_RUNS * sizeof(float), cudaMemcpyHostToDevice);
//
//     findSmallestElementInArray<<<1, 128>>>(d_a, NUM_RUNS, &d_result, 1); // 2 blocks (component size), 100 simulations but round up to 128
//
//     cudaMemcpy(h_a, d_a, NUM_RUNS * sizeof(float), cudaMemcpyDeviceToHost);
//
//     for (int i = 0; i < NUM_RUNS; i++) {
//         cout << h_a[i] << ", ";
//     }
//     cout <<endl << "Result = " << d_result << endl;
// }

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
