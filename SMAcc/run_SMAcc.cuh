#pragma once

// #include "common/sim_config.h"
// #include <unordered_set>
// #include "results/output_write.h"
#include "SMAcc_Tools.cuh"
#include "simulation_runner.h"

class SMAcc_Runner{
    public:
        static void run_SMAcc(simulation_config* input_config, network* model){
                sim_config SMAcc_config = {};

                convert_config(&SMAcc_config, input_config);

                // simulation_runner::simulate_gpu(model, &SMAcc_config);
                simulation_runner::simulate_gpu(model, &SMAcc_config);
            }   
};
