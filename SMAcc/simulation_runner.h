#pragma once
#include "allocations/cuda_allocator.h"
#include "engine/model_oracle.h"
#include "common/sim_config.h"


class simulation_runner
{
public:

    static void simulate_gpu(network* model, sim_config* config);
    static void simulate_gpu_jit(network* model, sim_config* config);
    static void simulate_cpu(const network* model, sim_config* config);

};