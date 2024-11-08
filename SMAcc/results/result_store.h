#pragma once
#include <mutex>

#include "../common/macro.h"
#include "../allocations/memory_allocator.h"
#include "../engine/Domain.h"
#include "limits.h"


struct node_results
{
    unsigned reached;

    unsigned total_steps;
    double total_time;

    double avg_steps() const
    {
        if(reached == 0) return 0.0;
        return static_cast<double>(total_steps) / static_cast<double>(reached);
    }

    double avg_time() const
    {
        if(reached == 0) return 0.0;
        return total_time / static_cast<double>(reached);

    }
};

struct variable_result
{
    double total_values;
    double max_value;

    double avg_max_value(const unsigned total_simulations) const
    {
        if(total_simulations == 0) return 0.0;
        return total_values / static_cast<double>(total_simulations);
    }
};

struct result_pointers
{
private:
    const bool owns_pointers_;
    void* source_p_;
public:
    explicit result_pointers(const bool owns_pointers,
        void* free_p,
        node_results* nodes,
        variable_result* variables,
        int threads,
        unsigned total_simulations);

    node_results* nodes = nullptr;
    variable_result* variables = nullptr;
    int simulations_per_thread = 0;
    int threads = 0;
    unsigned total_simulations = 0;

    unsigned sim_per_thread() const
    {
        return static_cast<unsigned>(ceilf(static_cast<float>(this->total_simulations) / static_cast<float>(threads)));
    }

    void free_internals() const;
};

class memory_allocator;


class result_store
{
    friend struct state;
private:
    bool is_cuda_;

    unsigned simulations_;
    unsigned node_count_;
    unsigned variables_count_;
    int n_parallelism;


    node_results* node_p_ = nullptr;
    variable_result* variable_p_ = nullptr;


    size_t total_data_size() const;

public:
    explicit result_store(
        unsigned total_sim,
        unsigned variables,
        unsigned node_count,
        int n_parallelism,
        memory_allocator* helper);

    result_pointers load_results() const;
    void clear() const;

    //This must be in .h for RDC=false to be used.
    CPU GPU void write_output(const unsigned idx,  const state* sim) const
    {
        const int offset = static_cast<int>(this->node_count_ * idx);
        for (int i = 0; i < sim->models.size; ++i)
        {
            const int index = offset + sim->models.store[i]->id - 1;
            this->node_p_[index].reached++;
            this->node_p_[index].total_steps += sim->steps;
            this->node_p_[index].total_time += sim->global_time;
        }

        const int var_offset = static_cast<int>(this->variables_count_ * idx);
        for (int i = 0, j = 0; i < sim->variables.size; ++i)
        {
            if(!sim->variables.store[i].should_track) continue;
            const int index = var_offset + j++;
            this->variable_p_[index].total_values += sim->variables.store[i].max_value;
            this->variable_p_[index].max_value = fmax(
                this->variable_p_[index].max_value,
                sim->variables.store[i].max_value);
        }
    }
};