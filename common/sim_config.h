
#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

#define SHARED_MEMORY_PR_THREAD 32

#include "macro.h"
struct io_paths;
struct output_properties;
#include "io_paths.h"

struct sim_config
{
    //simulation setup
    unsigned int blocks = 1;
    unsigned int threads = 1;
    unsigned int cpu_threads = 1;
    unsigned int sim_pr_thread = 1;
    unsigned int simulation_repetitions = 1;
    unsigned long long seed = 1;
    int write_mode = 0;
    bool use_max_steps = true;
    unsigned int max_sim_steps = 1;
    double max_global_progression = 1;
    bool verbose = false;
    enum pretty_print
    {
        no_print = 0,
        print_model = 1,
        print_reduction = 2
    } model_print_mode = no_print;
    
    enum device_opt
    {
        device,
        host,
        both
    } sim_location = device;
    
    //model parameters (setup using function)
    bool use_shared_memory = false;
    bool use_jit = false;
    bool use_pn = false;
    unsigned max_backtrace_depth = 1;
    unsigned max_expression_depth = 1;
    unsigned max_edge_fanout = 0;
    unsigned tracked_variable_count = 1;
    unsigned variable_count = 1;
    unsigned network_size = 1;
    unsigned node_count = 0;
    unsigned initial_urgent = 0;
    unsigned initial_committed = 0;
    
    //paths
    io_paths* paths = nullptr;

    output_properties* properties = nullptr;
    double alpha = 0.005;
    double epsilon = 0.005;
    int upscale = 1;
    
    //pointers
    void* cache = nullptr;
    curandState* random_state_arr = nullptr;
    
    size_t total_simulations() const
    {
        return sim_location == device
            ? sim_pr_thread * blocks * threads
            : sim_pr_thread * cpu_threads;
    }

    bool can_use_cuda_shared_memory(const size_t model_size) const
    {
        return (static_cast<size_t>(this->threads) * SHARED_MEMORY_PR_THREAD) > (model_size);
    }
};

#endif