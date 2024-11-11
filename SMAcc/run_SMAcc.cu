//This file contains everything we will need to run SMAcc
// It is seperate from other files, as to not include our changes,
// but be faithful to the original smacc implementation
//Their github repo is found at: https://github.com/Baksling/P7-SMAcc

#include "run_SMAcc.cuh"

void run_SMAcc(simulation_config* input_config, network* model) {

    sim_config SMAcc_config = {};

    convert_config(&SMAcc_config, input_config);

    // simulation_runner::simulate_gpu(model, &SMAcc_config);
    simulation_runner::simulate_gpu(model, &SMAcc_config);

}
