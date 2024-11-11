#pragma once

#include "../../network_optimization/visitor.h"
#include "../../engine/model_oracle.h"

class model_count_visitor : public visitor
{
private:
    model_size counter_;
public:
    void visit(network* a) override;
    void visit(node* n) override;
    void visit(edge* e) override;
    void visit(constraint* c) override;
    void visit(clock_var* cv) override;
    void visit(update* u) override;
    void visit(expr* ex) override;
    void clear() override;

    model_size get_model_size() const { return counter_; }
};