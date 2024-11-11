#include "simulation_runner.h"

using namespace std::chrono;

model_oracle* cuda_allocate(network* model, const sim_config* config,
    memory_allocator* allocator,
    size_t* fast_memory,
    cudaFuncCache* sm_cache)
{

    cuda_allocator av = cuda_allocator(allocator);

    if(config->use_shared_memory)
    {
        model_count_visitor count_visitor = model_count_visitor();
        count_visitor.visit(model);
        model_size size = count_visitor.get_model_size();

        memory_alignment_visitor alignment_visitor = memory_alignment_visitor();
        const model_oracle oracle = alignment_visitor.align(model, size, allocator);

        *fast_memory = oracle.model_counter.total_memory_size();
        *sm_cache = cudaFuncCachePreferEqual;
        return av.allocate_oracle(&oracle);
    }

    *fast_memory = 0;
    *sm_cache = cudaFuncCachePreferL1;
    return av.allocate_model(model);
}

void simulation_runner::simulate_gpu(network* model, sim_config* config)
{
    memory_allocator allocator = memory_allocator(true);

    const size_t n_parallelism = static_cast<size_t>(config->blocks)*config->threads;
    const size_t total_simulations = config->total_simulations();
    size_t fast_memory = 0;
    cudaFuncCache sm_cache;

    const model_oracle* oracle_d = cuda_allocate(model, config,
        &allocator,
        &fast_memory,
        &sm_cache);

    CUDA_CHECK(allocator.allocate_cuda(&config->cache, n_parallelism*thread_heap_size(config)));
    CUDA_CHECK(allocator.allocate_cuda(&config->random_state_arr, n_parallelism*sizeof(curandState)));

    sim_config* config_d = nullptr;
    CUDA_CHECK(allocator.allocate(&config_d, sizeof(sim_config)));
    CUDA_CHECK(cudaMemcpy(config_d, config, sizeof(sim_config), cudaMemcpyHostToDevice));


    const result_store store = result_store(
        static_cast<unsigned>(total_simulations),
        config->tracked_variable_count,
        config->node_count,
        static_cast<int>(n_parallelism),
        &allocator);

    result_store* store_d = nullptr;
    CUDA_CHECK(allocator.allocate(&store_d, sizeof(result_store)));
    CUDA_CHECK(cudaMemcpy(store_d, &store, sizeof(result_store), cudaMemcpyHostToDevice));

    output_writer writer = output_writer(config, model);

    CUDA_CHECK(cudaDeviceSetCacheConfig(sm_cache));

    if(config->verbose) std::cout << "GPU simulation started\n";
    const steady_clock::time_point global_start = steady_clock::now();
    for (unsigned r = 0; r < config->simulation_repetitions; ++r)
    {
        const steady_clock::time_point local_start = steady_clock::now();
        simulator_gpu_kernel<<<config->blocks, config->threads, fast_memory>>>(oracle_d, store_d, config_d);
        cudaDeviceSynchronize();
        const cudaError status = cudaPeekAtLastError();
        if(status != cudaSuccess)
            throw std::runtime_error("An error was encountered while running simulation. Error: " + std::to_string(status) + ".\n" );

        writer.write(
            &store,
            std::chrono::duration_cast<milliseconds>(steady_clock::now() - local_start));
    }
    if(config->verbose) std::cout << "GPU simulation finished\n";

    writer.write_summary(std::chrono::duration_cast<milliseconds>(steady_clock::now() - global_start));

    allocator.free_allocations();
}

