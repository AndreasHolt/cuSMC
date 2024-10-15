#pragma once
#include "../common/macro.h"
#include "Domain.h"

struct model_size
{
    unsigned network_size = 0;
    unsigned nodes = 0; 
    unsigned edges = 0; 
    unsigned constraints = 0;
    unsigned updates = 0;
    unsigned variables = 0;
    unsigned expressions = 0;

    CPU GPU size_t total_memory_size() const;

    bool operator==(const model_size& rhs) const
    {
        return network_size == rhs.network_size && nodes == rhs.nodes
        && edges == rhs.edges
        && constraints == rhs.constraints
        && updates == rhs.updates
        && variables == rhs.variables
        && expressions == rhs.expressions;
    }
};

class model_oracle
{
public:
    model_oracle(void* point, const model_size& model_count)
    {
        this->initial_point = point;
        this->point = point;
        this->model_counter = model_count;
    }
    
    void* initial_point;
    void* point;
    model_size model_counter;

    template<typename T>
    CPU GPU T* get_diff(void* p1, T* p2, char* source) const;

    CPU GPU network* network_point() const;
    CPU GPU node** network_nodes_point() const;
    CPU GPU node* node_point() const;
    CPU GPU edge* edge_point() const;
    CPU GPU constraint* constraint_point() const;
    CPU GPU update* update_point() const;
    CPU GPU expr* expression_point() const;
    CPU GPU clock_var* variable_point() const;

    GPU network* move_to_shared_memory(char* shared_mem, const int threads) const;
};

template <typename T>
CPU GPU T* model_oracle::get_diff(void* p1, T* p2, char* source) const
{
    const char* c1 = static_cast<char*>(p1);
    const char* c2 = static_cast<char*>(static_cast<void*>(p2));

    return static_cast<T*>(static_cast<void*>(&source[(c2 - c1)]));
}
