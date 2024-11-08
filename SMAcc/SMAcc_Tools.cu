#include "common/sim_config.h"

#include "../simulation/simulation_config.h"
#include "results/result_store.h"

//Based on https://github.com/Baksling/P7-SMAcc/blob/main/src2/Cuda/main.cu
void convert_config(sim_config* config, const simulation_config* in_config) {
    //This file creates a simple baseline config


    config->seed = in_config->seed;
    config->blocks = in_config->blocks;
    config->threads = in_config->blocks;
    config->cpu_threads = in_config->cpu_threads;
    config->sim_pr_thread = in_config->sim_pr_thread;
    config->simulation_repetitions = in_config->simulation_repetitions;
    config->write_mode = in_config->write_mode;
    config->use_max_steps = in_config->use_max_steps;
    config->max_sim_steps = in_config->max_sim_steps;
    config->max_global_progression = in_config->max_global_progression;
    config->verbose = false;


    // Cannot be converted as simulation_config does not have a querystring.
    // However it's use case is already applied
    config->paths->query = "";


    config->paths->output_path = "./output";
    config->sim_location = sim_config::device;
    config->upscale = in_config->upscale;

    config->model_print_mode = sim_config::no_print;

    config->use_shared_memory = true; //Attempt to move simulation to shared mem
    config->use_jit = false; //Use Just in time
    config->use_pn = false;  //Convert to polish notation. Cannot be used with shared mem

    double epsilon = in_config->epsilon;
    double alpha = 2 * exp((-2.0) * static_cast<double>(config->total_simulations()) * pow(epsilon,2));

    //Assign to config object
    config->alpha = alpha;
    config->epsilon = epsilon;

    config->max_expression_depth = in_config->max_expression_depth;
    config->max_edge_fanout = in_config->max_edge_fanout;
    config->tracked_variable_count = in_config->tracked_variable_count;
    config->variable_count = in_config->variable_count;
    config->network_size = in_config->network_size;
    config->node_count = in_config->node_count;
    config->initial_urgent = in_config->initial_urgent;
    config->initial_committed = in_config->initial_committed;
    config->cache = in_config->cache;
    config->random_state_arr = in_config->random_state_arr;

    //Copy properties
    config->properties->node_map = in_config->properties->node_map;
    config->properties->node_names = in_config->properties->node_names;
    config->properties->node_network = in_config->properties->node_network;
    config->properties->post_optimisation_start = in_config->properties->post_optimisation_start;
    config->properties->pre_optimisation_start = in_config->properties->pre_optimisation_start;
    config->properties->template_names = in_config->properties->template_names;
    config->properties->variable_names = in_config->properties->variable_names;

    //Copy paths
    config->paths->model_path = in_config->paths->model_path;
    config->paths->model_path = in_config->paths->output_path;
    config->paths->query = in_config->paths->query;
}