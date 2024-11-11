#include "../../include/common_macros.h"
#include "../../engine/Domain.cu"
// #include "../../engine/model_oracle.h"
#include "../../engine/model_oracle.cu"
#include "../common/sim_config.h"
#include "../results/result_store.h"
#include "device_launch_parameters.h"

CPU GPU size_t thread_heap_size(const sim_config* config);
CPU GPU double determine_progress(const node* node, state* state);
CPU GPU inline bool can_progress(const node* n);

CPU GPU inline bool is_winning_process(
    const double local_progress,
    const double min_progression_time,
    const unsigned epsilon,
    const unsigned max_epsilon,
    const state* sim_state,
    const node* current);

#define NO_PROCESS (-1)
#define IS_NO_PROCESS(x) ((x) < 0)
CPU GPU int progress_sim(state* sim_state, const sim_config* config);

CPU GPU edge* pick_next_edge_stack(const arr<edge>& edges, state* state);

CPU GPU void simulate_automata(
    const unsigned idx,
    const network* model,
    const result_store* output,
    const sim_config* config);

__global__ void simulator_gpu_kernel(
    const model_oracle* oracle,
    const result_store* output,
    const sim_config* config);