#include "visitor.h"

bool visitor::has_visited(const void* p)
{
    if(visit_set_.count(p)) return true;
    visit_set_.insert(p);
    return false;
}

void visitor::accept(const network* a, visitor* v)
{
    v->visit_set_.clear();
    for (int i = 0; i < a->automatas.size; ++i)
    {
        v->visit(a->automatas.store[i]);
    }

    for (int i = 0; i < a->variables.size; ++i)
    {
        v->visit(&a->variables.store[i]);
    }
}

void visitor::accept(const node* n, visitor* v)
{
    //The order of these visits are important, DO NOT CHANGE!
    for (int i = 0; i < n->invariants.size; ++i)
    {
        v->visit(&n->invariants.store[i]);
    }
    
    for (int i = 0; i < n->edges.size; ++i)
    {
        v->visit(&n->edges.store[i]);
    }

    for (int i = 0; i < n->edges.size; ++i)
    {
        v->visit(n->edges.store[i].dest);
    }
    
    v->visit(n->lamda);
}

void visitor::accept(const edge* e, visitor* v)
{
    // v->visit(e->dest);
    
    for (int i = 0; i < e->guards.size; ++i)
    {
        v->visit(&e->guards.store[i]);
    }

    for (int i = 0; i < e->updates.size; ++i)
    {
        v->visit(&e->updates.store[i]);
    }

    v->visit(e->weight);
}

void visitor::accept(const constraint* c, visitor* v)
{
    if(c->operand == constraint::compiled_c) return; //nothing of value
    if(!c->uses_variable)
    {
        v->visit(c->value);
    }
    v->visit(c->expression);
}

void visitor::accept(const clock_var* c, visitor* v)
{
    return;
}

void visitor::accept(const update* u, visitor* v)
{
    v->visit(u->expression);
}

void visitor::accept(const expr* ex, visitor* v)
{
    if(ex->left != nullptr)
    {
        v->visit(ex->left);
    }

    if(ex->right != nullptr)
    {
        v->visit(ex->right);
    }

    if(ex->operand == expr::conditional_ee && ex->conditional_else != nullptr)
    {
        v->visit(ex->conditional_else);
    }
}

void visitor::clear()
{
    this->visit_set_.clear();
}








