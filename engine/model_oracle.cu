#include "model_oracle.h"

CPU GPU size_t model_size::total_memory_size() const
{
    return  sizeof(network)
        +   sizeof(void*) * this->network_size
        +   sizeof(node) * this->nodes
        +   sizeof(edge) * this->edges
        +   sizeof(constraint) * this->constraints
        +   sizeof(update) * this->updates
        +   sizeof(expr) * this->expressions
        +   sizeof(clock_var) * this->variables;
}

CPU GPU network* model_oracle::network_point() const
{
    return static_cast<network*>(point);
}

CPU GPU node** model_oracle::network_nodes_point() const
{
    void* p = &network_point()[1];
    return static_cast<node**>(p);
}

CPU GPU node* model_oracle::node_point() const
{
    void* p = &network_nodes_point()[model_counter.network_size];
    return static_cast<node*>(p);
}

CPU GPU edge* model_oracle::edge_point() const
{
    void* p = &node_point()[model_counter.nodes];
    return static_cast<edge*>(p);
}

CPU GPU constraint* model_oracle::constraint_point() const
{
    void* p = &edge_point()[model_counter.edges];
    return static_cast<constraint*>(p);
}

CPU GPU update* model_oracle::update_point() const
{
    void* p = &constraint_point()[model_counter.constraints];
    return static_cast<update*>(p);
}

CPU GPU expr* model_oracle::expression_point() const
{
    void* p = &update_point()[model_counter.updates];
    return static_cast<expr*>(p);
}

CPU GPU clock_var* model_oracle::variable_point() const
{
    void* p = &expression_point()[model_counter.expressions];
    return static_cast<clock_var*>(p);
}


GPU network* model_oracle::move_to_shared_memory(char* shared_mem, const int threads) const
{
    size_t* wide_shared_memory = static_cast<size_t*>(static_cast<void*>(shared_mem));
    const size_t size = this->model_counter.total_memory_size() / sizeof(size_t);

    for (size_t i = 0; i < size; i += threads)
    {
        const size_t idx = i + threadIdx.x;
        if(!(idx < size)) continue;
        wide_shared_memory[idx] = static_cast<size_t*>(this->point)[idx];
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.nodes; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(idx >= static_cast<int>(this->model_counter.nodes)) continue;

        node* n = get_diff<node>(this->point, &this->node_point()[idx], shared_mem);

        n->edges.store = get_diff<edge>(this->initial_point, n->edges.store, shared_mem);
        n->invariants.store = get_diff<constraint>(this->initial_point, n->invariants.store, shared_mem);
        
        n->lamda = get_diff(this->initial_point, n->lamda, shared_mem);
    }
    cuda_SYNCTHREADS();


    for (unsigned i = 0; i < this->model_counter.edges; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(!(idx < static_cast<int>(this->model_counter.edges))) continue;
        
        edge* e = get_diff<edge>(this->point, &this->edge_point()[idx], shared_mem);

        e->dest = get_diff<node>(this->initial_point, e->dest, shared_mem);
        e->guards.store  = get_diff<constraint>(this->initial_point, e->guards.store, shared_mem);
        e->updates.store = get_diff<update>(this->initial_point, e->updates.store, shared_mem);
        e->weight = get_diff<expr>(this->initial_point, e->weight, shared_mem);
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.constraints; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(!(idx < static_cast<int>(this->model_counter.constraints))) continue;

        constraint* con = get_diff<constraint>(this->point, &this->constraint_point()[idx], shared_mem);

        con->expression = get_diff<expr>(this->initial_point, con->expression, shared_mem);
        if(!con->uses_variable && con->operand != constraint::compiled_c)
            con->value = get_diff<expr>(this->initial_point, con->value, shared_mem);
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.updates; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(!(idx < static_cast<int>(this->model_counter.updates))) continue;

        update* u = get_diff<update>(this->point, &this->update_point()[idx], shared_mem);
        
        u->expression = get_diff(this->initial_point, u->expression, shared_mem);
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.expressions; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(!(idx < static_cast<int>(this->model_counter.expressions))) continue;

        expr* ex = get_diff<expr>(this->point, &this->expression_point()[idx], shared_mem);

        if(ex->left != nullptr)
            ex->left = get_diff<expr>(this->initial_point, ex->left, shared_mem);
        if(ex->right != nullptr)
            ex->right = get_diff<expr>(this->initial_point, ex->right, shared_mem);
        if(ex->operand == ex->conditional_ee && ex->conditional_else != nullptr)
            ex->conditional_else = get_diff<expr>(this->initial_point, ex->conditional_else, shared_mem);
    }
    cuda_SYNCTHREADS();

    for (unsigned i = 0; i < this->model_counter.network_size; i += threads)
    {
        const int idx = static_cast<int>(i + threadIdx.x);
        if(idx >= static_cast<int>(this->model_counter.network_size)) continue;

        node** nn = get_diff<node*>(this->point, this->network_nodes_point(), shared_mem);
        nn[idx] = get_diff<node>(this->initial_point, nn[idx], shared_mem);
    }
    cuda_SYNCTHREADS();
    
    if(threadIdx.x == 0)
    {
        network* n = get_diff(this->point, this->network_point(), shared_mem);

        n->automatas.store = get_diff(this->point, this->network_nodes_point(), shared_mem);
        n->variables.store = get_diff(this->point, this->variable_point(), shared_mem);
    }
    cuda_SYNCTHREADS();

    return static_cast<network*>(static_cast<void*>(shared_mem));
}
