#pragma once

#include "common/sim_config.h"

#include "simulation_runner.h"
#include "results/output_write.h"
#include "allocations/cuda_allocator.h"
#include "allocations/memory_allocator.h"
#include "engine/automata_engine.cuh"
#include "../include/common_macros.h"
#include "common/thread_pool.h"
#include "visitors/model_count_visitor.h"

class simulation_runner
{
public:

    static void simulate_gpu(network* model, sim_config* config);

};