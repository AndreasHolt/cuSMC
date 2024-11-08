#include "cuda_allocator.h"

expr* cuda_allocator::move_expr(const expr* ex)
{
    if(ex == nullptr) return nullptr;

    expr* exp_d = nullptr;
    if(ex->operand == expr::pn_compiled_ee)
    {
        CUDA_CHECK(this->allocator_->allocate(&exp_d, sizeof(expr)*ex->length));
        CUDA_CHECK(cudaMemcpy(exp_d, ex, sizeof(expr)*ex->length, cudaMemcpyHostToDevice));
    }
    else
    {
        CUDA_CHECK(this->allocator_->allocate(&exp_d, sizeof(expr)));
        this->allocate_expr(ex, exp_d);
    }
    return exp_d;
}

model_oracle* cuda_allocator::allocate_model(const network* model)
{
    network* dest = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&dest, sizeof(network)));

    this->allocate_network(model, dest);

    model_oracle* oracle_d = nullptr;
    const model_oracle oracle = model_oracle(dest, {});
    this->allocator_->allocate(&oracle_d, sizeof(model_oracle));
    CUDA_CHECK(cudaMemcpy(oracle_d, &oracle, sizeof(model_oracle), cudaMemcpyHostToDevice));

    return oracle_d;
}

void cuda_allocator::allocate_network(const network* source, network* dest)
{
    const size_t network_size = sizeof(void*)*source->automatas.size;
    node* node_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&node_store, sizeof(node)*source->automatas.size));
    node** local_network = static_cast<node**>(malloc(network_size));

    for (int i = 0; i < source->automatas.size; ++i)
    {
        node* n = source->automatas.store[i];
        if(this->multi_model_resolver_.count(n))
        {
            local_network[i] = this->multi_model_resolver_.at(n);
        }
        else
        {
            this->allocate_node(source->automatas.store[i], &node_store[i]);
            local_network[i] = &node_store[i];
            this->multi_model_resolver_.insert(std::pair<const node*, node*>(n, &node_store[i]));
        }
    }

    node** network_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&network_store, network_size));
    CUDA_CHECK(cudaMemcpy(network_store, local_network, network_size, cudaMemcpyHostToDevice));

    clock_var* var_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&var_store, sizeof(clock_var)*source->variables.size));

    for (int i = 0; i < source->variables.size; ++i)
    {
        allocate_clock(&source->variables.store[i], &var_store[i]);
    }

    const network temp = {
        arr<node*>{ network_store, source->automatas.size },
        arr<clock_var>{ var_store, source->variables.size },
    };

    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(network), cudaMemcpyHostToDevice));
}

void cuda_allocator::allocate_node(const node* source, node* dest)
{
    if(this->circular_ref_.count(source)) return;
    this->circular_ref_.insert(std::pair<const node*, node*>(source, dest));

    edge* edge_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&edge_store, sizeof(edge)*source->edges.size));
    for (int i = 0; i < source->edges.size; ++i)
    {
        allocate_edge(&source->edges.store[i], &edge_store[i]);
    }

    constraint* invariant_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&invariant_store, sizeof(constraint)*source->invariants.size));
    for (int i = 0; i < source->invariants.size; ++i)
    {
        allocate_constraint(&source->invariants.store[i], &invariant_store[i]);
    }

    expr* exp_d = move_expr(source->lamda);
    // expr* exp_d = nullptr;
    // CUDA_CHECK(this->allocator_->allocate(&exp_d, sizeof(expr)));
    // this->allocate_expr(source->lamda, exp_d);


    const node temp = node{
        source->id,
        source->type,
        exp_d,
        arr<edge>{edge_store, source->edges.size},
        arr<constraint>{invariant_store, source->invariants.size},
    };

    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(node), cudaMemcpyHostToDevice));
}

void cuda_allocator::allocate_edge(const edge* source, edge* dest)
{
    //Handle node circular reference.
    node* node_dest = nullptr;
    if(this->circular_ref_.count(source->dest) == 0)
    {
        CUDA_CHECK(this->allocator_->allocate(&node_dest, sizeof(node)));
        this->allocate_node(source->dest, node_dest);
    }
    else
    {
        node_dest = this->circular_ref_[source->dest];
    }

    constraint* guard_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&guard_store, sizeof(constraint)*source->guards.size));
    for (int i = 0; i < source->guards.size; ++i)
    {
        allocate_constraint(&source->guards.store[i], &guard_store[i]);
    }

    update* update_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&update_store, sizeof(update)*source->updates.size));
    for (int i = 0; i < source->updates.size; ++i)
    {
        allocate_update(&source->updates.store[i], &update_store[i]);
    }

    expr* exp_d = move_expr(source->weight);
    // expr* exp_d = nullptr;
    // CUDA_CHECK(this->allocator_->allocate(&exp_d, sizeof(expr)));
    // this->allocate_expr(source->weight, exp_d);

    const edge temp = edge{
        source->channel,
        exp_d,
        node_dest,
        arr<constraint>{ guard_store, source->guards.size },
        arr<update>{ update_store, source->updates.size }
    };

    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(edge), cudaMemcpyHostToDevice));

}

void cuda_allocator::allocate_constraint(const constraint* source, constraint* dest)
{
    constraint temp{};
    temp.operand = source->operand;
    temp.uses_variable = source->uses_variable;

    if(source->operand == constraint::compiled_c)
    {
        temp.compile_id = source->compile_id;
    }
    else if(source->uses_variable)
    {
        temp.variable_id = source->variable_id;
    }
    else
    {
        temp.value  = move_expr(source->value);
        // expr* left = nullptr;
        // CUDA_CHECK(this->allocator_->allocate(&left, sizeof(expr)));
        // this->allocate_expr(source->value, left);
        // temp.value = left;
    }

    temp.expression = move_expr(source->expression);
    // expr* right = nullptr;
    // CUDA_CHECK(this->allocator_->allocate(&right, sizeof(expr)));
    // this->allocate_expr(source->expression, right);
    // temp.expression = right;

    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(constraint), cudaMemcpyHostToDevice));
}

void cuda_allocator::allocate_update(const update* source, update* dest)
{
    expr* right = move_expr(source->expression);
    // expr* right = nullptr;
    // CUDA_CHECK(this->allocator_->allocate(&right, sizeof(expr)));
    // this->allocate_expr(source->expression, right);

    const update temp{
        source->variable_id,
        right
    };
    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(update), cudaMemcpyHostToHost));
}

// ReSharper disable once CppMemberFunctionMayBeStatic
void cuda_allocator::allocate_clock(const clock_var* source, clock_var* dest) const
{
    CUDA_CHECK(cudaMemcpy(dest, source, sizeof(clock_var), cudaMemcpyHostToDevice));
}

void cuda_allocator::allocate_expr(const expr* source, expr* dest)
{
    if(IS_LEAF(source->operand) || source->operand == expr::compiled_ee)
    {
        CUDA_CHECK(cudaMemcpy(dest, source, sizeof(expr), cudaMemcpyHostToDevice));
        return;
    }

    expr* left = nullptr;
    if(source->left != nullptr)
    {
        CUDA_CHECK(this->allocator_->allocate(&left, sizeof(expr)));
        this->allocate_expr(source->left, left);
    }

    expr* right = nullptr;
    if(source->right != nullptr)
    {
        CUDA_CHECK(this->allocator_->allocate(&right, sizeof(expr)));
        this->allocate_expr(source->right, right);
    }

    expr* else_branch = nullptr;
    if(source->operand == expr::conditional_ee && source->conditional_else != nullptr)
    {
        CUDA_CHECK(this->allocator_->allocate(&else_branch, sizeof(expr)));
        this->allocate_expr(source->conditional_else, else_branch);
    }

    expr temp{};
    temp.left = left;
    temp.right = right;
    temp.operand = source->operand;
    temp.value = source->value; //biggest bit repressentation, so just move this.

    //this shouldn't be necessary, but safety is nr. 1 priority.
    if     (source->operand == expr::literal_ee)        temp.value            = source->value;
    else if(source->operand == expr::clock_variable_ee) temp.variable_id      = source->variable_id;
    else if(source->operand == expr::conditional_ee)    temp.conditional_else = else_branch;

    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(expr), cudaMemcpyHostToDevice));
}

model_oracle* cuda_allocator::allocate_oracle(const model_oracle* oracle) const
{
    void* point = nullptr;
    const size_t size = oracle->model_counter.total_memory_size();
    CUDA_CHECK(this->allocator_->allocate_cuda(&point, size));
    CUDA_CHECK(cudaMemcpy(point, oracle->point, size, cudaMemcpyHostToDevice));

    model_oracle temp = model_oracle(point, oracle->model_counter);
    temp.initial_point = oracle->initial_point;

    model_oracle* oracle_d = nullptr;
    CUDA_CHECK(this->allocator_->allocate_cuda(&oracle_d, sizeof(model_oracle)));
    CUDA_CHECK(cudaMemcpy(oracle_d, &temp, sizeof(model_oracle), cudaMemcpyHostToDevice));

    return oracle_d;
}

void cuda_allocator::clear()
{
    this->circular_ref_.clear();
    this->multi_model_resolver_.clear();
}