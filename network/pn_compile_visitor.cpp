#include "pn_compile_visitor.h"

int pn_compile_visitor::estimate_pn_lenght(const expr* ex)
{
    if(IS_LEAF(ex->operand)) return 1;
    
    const int left  = ex->left  != nullptr ? estimate_pn_lenght(ex->left) : 0;
    const int right = ex->right != nullptr ? estimate_pn_lenght(ex->right) : 0;

    if(ex->operand == expr::conditional_ee)
    {
        constexpr int cond_jump = 1; //needs padding after condition for jump
        constexpr int else_jump = 2;  //needs padding after else case, as to push true and skip.
        const int else_ = estimate_pn_lenght(ex->conditional_else);

        return left + cond_jump + else_ + else_jump + right;
    }
    
    return (left + right) + 1;
}

void pn_compile_visitor::compile_expr(expr** ex_p)
{
    const expr* ex = *ex_p;
    if(IS_LEAF(ex->operand)) return;
    
    const int size = estimate_pn_lenght(ex) + 1;
    expr* array = static_cast<expr*>(malloc(sizeof(expr) * size));

    expr init = {expr::pn_compiled_ee, nullptr,nullptr, {0.0}};
    init.length = size;
    int index = 1;
    array[0] = init;

    pn_visitor(ex, array, &index);
    
    *ex_p = array;
    // if(ex->operand == expr::conditional_ee)
    // {
    //     printf("How tf did i get here. size: %d | index: %d, | 0in: %d | arr[0]: %d \n", size, index, (*ex_p)[0].operand, array[0].operand);
    // }
}

void pn_compile_visitor::pn_visitor(const expr* current, expr* array, int* index)
{
    if(current == nullptr) return;
    if(current->operand == expr::conditional_ee)
    {
        expr skip = {expr::pn_skips_ee, nullptr, nullptr, {0.0}};
        skip.length = estimate_pn_lenght(current->conditional_else) + 2; // + the two control structures of the else. 
        constexpr expr true_expr = { expr::literal_ee, nullptr, nullptr, {1.0}};

        pn_visitor(current->left, array, index); //compile condition into pn
        array[(*index)++] = skip; //add goto statement to true part of statement. If condition doesnt hold, it will goto false. 
        
        pn_visitor(current->conditional_else, array, index);
        array[(*index)++] = true_expr; //push true literal onto stack so the following condition will pass.
        skip.length = estimate_pn_lenght(current->right); //Add jump to after condition.
        array[(*index)++] = skip;

        pn_visitor(current->right, array, index);
        return;
    }
    
    pn_visitor(current->left, array, index);
    pn_visitor(current->right, array, index);
    array[(*index)++] = *current;
}

void pn_compile_visitor::visit(network* a)
{
    if(has_visited(a)) return;

    accept(a, this);
}

void pn_compile_visitor::visit(node* n)
{
    if(has_visited(n)) return;

    this->compile_expr(&n->lamda);
    
    accept(n, this);
}

void pn_compile_visitor::visit(edge* e)
{
    if(has_visited(e)) return;

    this->compile_expr(&e->weight);

    
    accept(e, this);
}

void pn_compile_visitor::visit(constraint* c)
{
    if(has_visited(c)) return;

    if(!c->uses_variable)
        this->compile_expr(&c->value);

    this->compile_expr(&c->expression);
    
    accept(c, this);
}

void pn_compile_visitor::visit(clock_var* cv)
{
    if(has_visited(cv)) return;
    
    accept(cv, this);
}

void pn_compile_visitor::visit(update* u)
{
    if(has_visited(u)) return;

    this->compile_expr(&u->expression);
    
    accept(u, this);
}

void pn_compile_visitor::visit(expr* ex)
{
    if(has_visited(ex)) return;

    // accept(ex, this);
}

void pn_compile_visitor::clear()
{
    visitor::clear();
}