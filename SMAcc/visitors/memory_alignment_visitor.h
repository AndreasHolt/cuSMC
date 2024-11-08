#pragma once

#include "visitor.h"
#include "domain_optimization_visitor.h"
#include "../allocations/memory_allocator.h"
#include "../engine/model_oracle.h"


class memory_alignment_visitor : public visitor
{
    model_size move_state_{};
    model_oracle oracle_{nullptr, {}};
    std::unordered_map<void*, void*> location_mapper_;

    void post_process() const;
public:
    model_oracle align(network* n, const model_size& model_m, memory_allocator* allocator);

    void visit(network* nn) override;
    void visit(node* n) override;
    void visit(edge* e) override;
    void visit(constraint* c) override;
    void visit(clock_var* cv) override ;
    void visit(update* u) override;
    void visit(expr* ex) override;

    void clear() override;
};