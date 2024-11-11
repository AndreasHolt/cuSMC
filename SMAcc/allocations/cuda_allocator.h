#pragma once
#include <unordered_map>

#include "memory_allocator.h"
#include "../../engine/Domain.h"
#include "../visitors/memory_alignment_visitor.h"

class cuda_allocator
{
    memory_allocator* allocator_;
    std::unordered_map<const node*, node*> multi_model_resolver_{};
    std::unordered_map<const node*, node*> circular_ref_{};

    expr* move_expr(const expr* ex);
public:
    explicit cuda_allocator(memory_allocator* allocator)
    {
        this->allocator_ = allocator;
        this->circular_ref_ = std::unordered_map<const node*, node*>();
    }

    //The one to rule them all!
    model_oracle* allocate_model(const network* model);

    void allocate_network(const network* source, network* dest);
    void allocate_node(const node* source, node* dest);
    void allocate_edge(const edge* source, edge* dest);
    void allocate_constraint(const constraint* source, constraint* dest);
    void allocate_update(const update* source, update* dest);
    void allocate_clock(const clock_var* source, clock_var* dest) const;
    void allocate_expr(const expr* source, expr* dest);

    model_oracle* allocate_oracle(const model_oracle* oracle) const;
    void clear();
};