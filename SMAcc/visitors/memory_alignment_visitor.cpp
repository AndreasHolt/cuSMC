#include "memory_alignment_visitor.h"

#include <iostream>

#define EXISTS(x)       \
    if(this->location_mapper_.count(x) == 0)                                      \
        throw std::runtime_error("Cannot find item at line: " + __LINE__) \

void memory_alignment_visitor::post_process() const
{
    edge* e_point = this->oracle_.edge_point();
    for (unsigned i = 0; i < this->oracle_.model_counter.edges; ++i)
    {
        edge* e = &e_point[i];
        EXISTS(e->dest);
        e->dest = static_cast<node*>(this->location_mapper_.at(e->dest));
    }
}

#define MATCH(x,y, name) \
    do{ \
    if((x) != (y)){   \
        std::cout << "mismatch " << (x) << ' ' << (y) << std::endl; \
        throw std::out_of_range(name); \
        } \
    } while(0)


model_oracle memory_alignment_visitor::align(network* n, const model_size& model_m, memory_allocator* allocator)
{
    this->location_mapper_.clear();
    void* memory;
    allocator->allocate_host(&memory, model_m.total_memory_size());
    this->oracle_ = model_oracle(memory, model_m );
    this->move_state_ = model_size{};

    visit(n);

    MATCH(model_m.network_size, move_state_.network_size, "network");
    MATCH(model_m.nodes, move_state_.nodes, "nodes");
    MATCH(model_m.edges, move_state_.edges, "edges");
    MATCH(model_m.constraints, move_state_.constraints, "cosntraints");
    MATCH(model_m.updates, move_state_.updates, "updates");
    MATCH(model_m.expressions, move_state_.expressions, "expressions");
    MATCH(model_m.variables, move_state_.variables, "variables");

    post_process();

    return this->oracle_;
}

void memory_alignment_visitor::visit(network* nn)
{
    if(has_visited(nn)) return;
    this->move_state_.network_size = nn->automatas.size;

    network* net = this->oracle_.network_point();
    *net = *nn;
    this->location_mapper_.insert(std::pair<void*, void*>(nn, net));

    net->automatas.store = this->oracle_.network_nodes_point();
    net->variables.store = this->oracle_.variable_point();

    accept(nn, this);

    for (int i = 0; i < net->automatas.size; ++i)
    {
        EXISTS(nn->automatas.store[i]);
        net->automatas.store[i] = static_cast<node*>(this->location_mapper_.at(nn->automatas.store[i]));
    }
}

void memory_alignment_visitor::visit(node* n)
{
    if(has_visited(n)) return;

    node* no = &this->oracle_.node_point()[this->move_state_.nodes++];
    *no = *n;
    this->location_mapper_.insert(std::pair<void*, void*>(n, no));

    //this assumes invariants are visited first, and then edges, and then destination nodes.
    no->invariants.store  = &this->oracle_.constraint_point()[this->move_state_.constraints];
    no->edges.store = &this->oracle_.edge_point()[this->move_state_.edges];
    accept(n, this);

    EXISTS(n->lamda);
    no->lamda = static_cast<expr*>(this->location_mapper_.at(n->lamda));
}

void memory_alignment_visitor::visit(edge* e)
{
    if(has_visited(e)) return;

    edge* ed = &this->oracle_.edge_point()[this->move_state_.edges++];
    *ed = *e;
    this->location_mapper_.insert(std::pair<void*, void*>(e, ed));

    ed->guards.store = &this->oracle_.constraint_point()[this->move_state_.constraints];
    ed->updates.store = &this->oracle_.update_point()[this->move_state_.updates];

    accept(e, this);

    EXISTS(ed->weight);
    ed->weight = static_cast<expr*>(this->location_mapper_.at(e->weight));
}

void memory_alignment_visitor::visit(constraint* c)
{
    if(has_visited(c)) return;

    constraint* co = &this->oracle_.constraint_point()[this->move_state_.constraints++];
    *co = *c;
    this->location_mapper_.insert(std::pair<void*, void*>(c, co));

    accept(c, this);

    if(!c->uses_variable && c->operand != constraint::compiled_c)
    {
        EXISTS(c->value);
        co->value = static_cast<expr*>(this->location_mapper_.at(c->value));
    }
    EXISTS(c->expression);
    co->expression = static_cast<expr*>(this->location_mapper_.at(c->expression));
}

void memory_alignment_visitor::visit(clock_var* cv)
{
    if(has_visited(cv)) return;

    clock_var* cov = &this->oracle_.variable_point()[this->move_state_.variables++];

    *cov = *cv;
    this->location_mapper_.insert(std::pair<void*, void*>(cv, cov));

    accept(cv, this);
}

void memory_alignment_visitor::visit(update* u)
{
    if(has_visited(u)) return;

    update* up = &this->oracle_.update_point()[this->move_state_.updates++];
    *up = *u;
    this->location_mapper_.insert(std::pair<void*, void*>(u, up));

    accept(u, this);

    EXISTS(u->expression);
    up->expression = static_cast<expr*>(this->location_mapper_.at(u->expression));
}

void memory_alignment_visitor::visit(expr* ex)
{
    if(has_visited(ex)) return;

    expr* exp = &this->oracle_.expression_point()[this->move_state_.expressions++];
    *exp = *ex;
    this->location_mapper_.insert(std::pair<void*, void*>(ex, exp));

    accept(ex, this);

    if(ex->left  != nullptr)
    {
        EXISTS(ex->left);
        exp->left  = static_cast<expr*>(this->location_mapper_.at(ex->left));
    }
    if(ex->right != nullptr)
    {
        EXISTS(ex->right);
        exp->right = static_cast<expr*>(this->location_mapper_.at(ex->right));
    }

    if(ex->operand == expr::conditional_ee && ex->conditional_else != nullptr)
    {
        EXISTS(ex->conditional_else);
        exp->conditional_else = static_cast<expr*>(this->location_mapper_.at(ex->conditional_else));
    }
}

void memory_alignment_visitor::clear()
{
    visitor::clear();
    location_mapper_.clear();
    oracle_ = {nullptr, {}};
    move_state_ = {};
}