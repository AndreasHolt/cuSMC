#include "model_count_visitor.h"

void model_count_visitor::visit(network* a)
{
    if(has_visited(a)) return;

    this->counter_.network_size = a->automatas.size;
    accept(a, this);
}

void model_count_visitor::visit(node* n)
{
    if(has_visited(n)) return;
    this->counter_.nodes++;

    accept(n, this);
}

void model_count_visitor::visit(edge* e)
{
    if(has_visited(e)) return;
    this->counter_.edges++;

    accept(e, this);
}

void model_count_visitor::visit(constraint* c)
{
    if(has_visited(c)) return;
    this->counter_.constraints++;

    accept(c, this);
}

void model_count_visitor::visit(clock_var* cv)
{
    if(has_visited(cv)) return;
    this->counter_.variables++;

    accept(cv, this);
}

void model_count_visitor::visit(update* u)
{
    if(has_visited(u)) return;
    this->counter_.updates++;

    accept(u, this);
}

void model_count_visitor::visit(expr* ex)
{
    if(has_visited(ex)) return;
    this->counter_.expressions++;

    accept(ex, this);
}

void model_count_visitor::clear()
{
    visitor::clear();
    counter_ = {};
}