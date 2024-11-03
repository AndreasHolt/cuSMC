#pragma once
#include "visitor.h"

class pn_compile_visitor final : public visitor
{
    static int estimate_pn_lenght(const expr* ex);
    static void compile_expr(expr** ex_p);
    static void pn_visitor(const expr* current, expr* array, int* index);
public:
    void visit(network* a) override;
    void visit(node* n) override;
    void visit(edge* e) override;
    void visit(constraint* c) override;
    void visit(clock_var* cv) override ;
    void visit(update* u) override;
    void visit(expr* ex) override;

    void clear() override;
};
