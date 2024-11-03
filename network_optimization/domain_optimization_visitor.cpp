#include "domain_optimization_visitor.h"
#include <list>

void domain_optimization_visitor::visit(network* a)
{
    if(has_visited(a)) return;

    for (int i = 0; i < a->variables.size; ++i)
    {
        clock_var* var = &a->variables.store[i];
        variables_clock_map_.insert(std::pair<int, bool>(var->id, var->rate > 0));
    }
    
    accept(a, this);
}

void domain_optimization_visitor::validate_invariants(const node* n)
{
    for (int i = 0; i < n->invariants.size; ++i)
    {
        const constraint& con = n->invariants.store[i];
        if(con.uses_variable)
        {
            this->contains_invalid_constraint_ = this->contains_invalid_constraint_ || expr_contains_clock(con.expression);
        }
    }
}

void domain_optimization_visitor::collect_node_data(node* n)
{
    this->node_map_.insert(std::pair<int,node*>(n->id, n));
    this->node_count_ = std::max(this->node_count_, static_cast<unsigned>(n->id));
    this->max_edge_fanout_ = std::max(this->max_edge_fanout_, static_cast<unsigned>(n->edges.size));
}

void domain_optimization_visitor::visit(node* n)
{
    if(has_visited(n)) return;
    
    collect_node_data(n);
    validate_invariants(n);
    
    if(use_model_reductions_)
    {
        reduce_constraint_set(&n->invariants);
    }
    
    if(this->is_goal(n->id))
        n->type = node::goal;

    accept(n, this);
}

void domain_optimization_visitor::visit(edge* e)
{
    if(has_visited(e)) return;
    
    compound_optimize_constraints(e);

    if(use_model_reductions_)
        reduce_constraint_set(&e->guards);
    
    accept(e, this);
}

void domain_optimization_visitor::visit(constraint* c)
{
    if(has_visited(c)) return;
    
    accept(c, this);
}

void domain_optimization_visitor::visit(clock_var* cv)
{
    if(has_visited(cv)) return;

    accept(cv, this);
}

void domain_optimization_visitor::visit(update* u)
{
    if(has_visited(u)) return;
    
    accept(u, this);
}

void domain_optimization_visitor::visit(expr* ex)
{
    if(has_visited(ex)) return;

    if(use_model_reductions_)
        reduce_expr(ex);
    
    this->max_expr_depth_ = std::max(this->max_expr_depth_, count_expr_depth(ex));
    // bool has_lock = false;
    // if(check_depth_lock_)
    // {
    //     has_lock = true;
    //     check_depth_lock_ = false;
    //     const unsigned max_depth = count_expr_depth(ex);
    //     this->max_expr_depth_ = std::max(this->max_expr_depth_, max_depth);
    // }
    //
    // accept(ex, this);
    //
    // if(has_lock) check_depth_lock_ = true;
}

void domain_optimization_visitor::clear()
{
    visitor::clear();
    this->node_count_ = 0;
    this->max_expr_depth_ = 0;
    this->max_edge_fanout_ = 0;
    this->check_depth_lock_ = true;
    this->variables_clock_map_.clear();
    this->check_depth_lock_ = true;
    this->node_map_.clear();
}

unsigned domain_optimization_visitor::get_max_expr_depth() const
{
    return max_expr_depth_;
}

bool domain_optimization_visitor::has_invalid_constraint() const
{
    return this->contains_invalid_constraint_;
}

unsigned domain_optimization_visitor::get_max_fanout() const
{
    return this->max_edge_fanout_;
}

unsigned domain_optimization_visitor::get_node_count() const
{
    return this->node_count_;
}

std::unordered_map<int, node*> domain_optimization_visitor::get_node_map() const
{
    return {this->node_map_};
}


bool domain_optimization_visitor::is_goal(const int node_id) const
{
    if (this->node_names_->count(node_id) == 0) return false;
    if(this->node_subsystems_map_->count(node_id) == 0)
        throw std::runtime_error("Node " + std::to_string(node_id) + " does not belong to any subsystem");
    
    const int process_id = this->node_subsystems_map_->at(node_id);
    if(this->subsystem_names_->count(process_id) == 0)
        throw std::runtime_error("subsystem with id " + std::to_string(process_id) + " could not be found");

    
    const std::string process_name = this->subsystem_names_->at(process_id);
    const std::string node_name = this->node_names_->at(node_id);
    const std::string name = process_name + "." + node_name;
    
    return this->query_->count(name);
}

unsigned domain_optimization_visitor::count_expr_depth(const expr* ex)
{
    const unsigned conditional = ex->operand == expr::conditional_ee ? count_expr_depth(ex->conditional_else) : 0;
    const unsigned left = ex->left != nullptr ? count_expr_depth(ex->left) : 0;
    const unsigned right = ex->right != nullptr ? count_expr_depth(ex->right) : 0;

    const unsigned temp = (left > right ? left : right);
    return (conditional > temp ? conditional : temp) + 1;
}

expr* deepcopy_expr(expr* ex)
{
    if(ex == nullptr) return nullptr;

    expr* copy = new expr();
    copy->operand = ex->operand;
    copy->left = deepcopy_expr(ex->left);
    copy->right = deepcopy_expr(ex->right);
    if(ex->operand == expr::conditional_ee)
        copy->conditional_else = deepcopy_expr(ex->conditional_else);
    else if(ex->operand == expr::clock_variable_ee)
        copy->variable_id = ex->variable_id;
    else if(ex->operand == expr::literal_ee)
        copy->value = ex->value;
    else if(ex->operand == expr::compiled_ee)
        throw std::runtime_error("Cannot deepcopy a compiled expression");

    return copy;
}

void domain_optimization_visitor::compound_optimize_constraints(edge* e)
{
    std::list<constraint> con_lst;
    //add all guards from edge
    for (int i = 0; i < e->guards.size; ++i)
    {
        con_lst.push_back(e->guards.store[i]);
    }

    //go through each invariant
    for (int i = 0; i < e->dest->invariants.size; ++i)
    {
        constraint inv = e->dest->invariants.store[i];
        inv.expression = interleave_updates_in_expr(inv.expression, e->updates);
        
        //if invariant does not use variable, 
        if(!inv.uses_variable)
        {
            inv.value = interleave_updates_in_expr(inv.value, e->updates);        
            con_lst.push_back(inv);
            continue;   
        }
        
        bool any = false;
        for (int j = 0; j < e->updates.size; ++j)
        {
            const update& upd = e->updates.store[j];
            if(upd.variable_id != inv.variable_id) continue;
            constraint compound = {};
            compound.operand = inv.operand;
            compound.value = upd.expression;
            compound.uses_variable = false;
            compound.expression = inv.expression;
            
            any = true;
            con_lst.push_back(compound);
        }
        if(!any) con_lst.push_back(inv);
    }

    // if(con_lst.size() <= static_cast<size_t>(e->guards.size)) return;

    constraint* store = static_cast<constraint*>(malloc(sizeof(constraint)*con_lst.size()));
    int i = 0;
    for (const constraint& con : con_lst)
    {
        store[i++] = con;
    }
    e->guards = arr<constraint>{ store, i };
}

expr* domain_optimization_visitor::interleave_updates_in_expr(expr* ex, const arr<update>& updates)
{
    if(ex == nullptr) return nullptr;
    if(ex->operand == expr::clock_variable_ee)
    {
        for (int i = 0; i < updates.size; ++i)
        {
            if(ex->variable_id != updates.store[i].variable_id) continue;
            return deepcopy_expr(updates.store[i].expression);
        }

        return ex; //not affected by updates
    }
    if(ex->operand == expr::literal_ee)
        return ex;

    expr* copy = new expr();
    copy->operand = ex->operand;
    copy->left = interleave_updates_in_expr(ex->left, updates);
    copy->right = interleave_updates_in_expr(ex->right, updates);
    
    if(ex->operand == expr::conditional_ee)
        copy->conditional_else = interleave_updates_in_expr(ex->conditional_else, updates);
    
    return copy;
}

bool domain_optimization_visitor::expr_contains_clock(const expr* ex)
{
    if(ex->operand == expr::literal_ee) return false;
    if(ex->operand == expr::clock_variable_ee)
        return this->variables_clock_map_.at(ex->variable_id);

    const bool left = ex->left != nullptr ? expr_contains_clock(ex->left) : false;
    const bool right = ex->right != nullptr ? expr_contains_clock(ex->right) : false;
    const bool cond_else = ex->operand == expr::conditional_ee && ex->conditional_else != nullptr
                               ? expr_contains_clock(ex->conditional_else) : false;

    return left || right || cond_else;
}


inline bool domain_optimization_visitor::is_const_expr(const expr* ex)
{
    if(ex->operand == expr::literal_ee) return true;
    if(ex->operand == expr::clock_variable_ee) return false;
    if(ex->operand == expr::random_ee) return false;

    const bool left = ex->left != nullptr ? is_const_expr(ex->left) : true;
    const bool right = ex->right != nullptr ? is_const_expr(ex->right) : true;
    const bool cond = ex->operand == expr::conditional_ee ? is_const_expr(ex->conditional_else) : true;

    
    return cond && left && right;
}

bool domain_optimization_visitor::is_const_constraint(const constraint* con)
{
    if(con->uses_variable) return false;
    const bool left = is_const_expr(con->value);
    const bool right = is_const_expr(con->expression);
    return left && right;
}

bool domain_optimization_visitor::evaluate_const_constraint(const constraint* con)
{
    if(con->uses_variable) throw std::runtime_error("Cannot evaluate constraint as constant, if a variable is used");
    const double left = evaluate_const_expr(con->value);
    const double right = evaluate_const_expr(con->expression);
    
    switch(con->operand) {
        case constraint::less_equal_c: return left <= right;
        case constraint::less_c: return left < right;
        case constraint::greater_equal_c: return left >= right;
        case constraint::greater_c: return left > right;
        case constraint::equal_c: return abs(left - right) <= DBL_EPSILON;
        case constraint::not_equal_c: return abs(left - right) > DBL_EPSILON;
        case constraint::compiled_c: throw std::runtime_error("Cannot evaluate compiled constraint as constant.");
        default: throw std::runtime_error("unknown operand while evaluating const constraint");
    }

}

double domain_optimization_visitor::evaluate_const_expr(const expr* ex)
{
    double temp;
    switch (ex->operand)
    {
        case expr::literal_ee: return ex->value;
        case expr::clock_variable_ee: throw std::runtime_error("Tried evaluating variable expression as constant expression in optimizer.");
        case expr::random_ee: throw std::runtime_error("Tried evaluating random expression as constant expression in optimizer.");
        case expr::plus_ee: return evaluate_const_expr(ex->left) + evaluate_const_expr(ex->right);
        case expr::minus_ee: return evaluate_const_expr(ex->left) - evaluate_const_expr(ex->right);
        case expr::multiply_ee: return evaluate_const_expr(ex->left) * evaluate_const_expr(ex->right);
        case expr::division_ee:
            temp = evaluate_const_expr(ex->right);
            if(temp == 0.0) throw std::runtime_error("Found division by zero in const expression, while running optimizer.");
            return evaluate_const_expr(ex->left) / temp;
        case expr::power_ee: return pow(evaluate_const_expr(ex->left), evaluate_const_expr(ex->right));
        case expr::negation_ee: return -evaluate_const_expr(ex->left);
        case expr::sqrt_ee: return sqrt(evaluate_const_expr(ex->left));
        case expr::modulo_ee: return static_cast<int>(evaluate_const_expr(ex->left)) % static_cast<int>(evaluate_const_expr(ex->right));
        case expr::and_ee: return abs(evaluate_const_expr(ex->left)) > DBL_EPSILON && abs(evaluate_const_expr(ex->right)) > DBL_EPSILON; 
        case expr::or_ee: return abs(evaluate_const_expr(ex->left)) > DBL_EPSILON || abs(evaluate_const_expr(ex->right)) > DBL_EPSILON; 
        case expr::less_equal_ee: return evaluate_const_expr(ex->left) <= evaluate_const_expr(ex->right);
        case expr::greater_equal_ee: return evaluate_const_expr(ex->left) >= evaluate_const_expr(ex->right);
        case expr::less_ee: return evaluate_const_expr(ex->left) < evaluate_const_expr(ex->right);
        case expr::greater_ee: return evaluate_const_expr(ex->left) > evaluate_const_expr(ex->right);
        case expr::equal_ee: return abs(evaluate_const_expr(ex->left) - evaluate_const_expr(ex->right)) <= DBL_EPSILON;
        case expr::not_equal_ee: return abs(evaluate_const_expr(ex->left) - evaluate_const_expr(ex->right)) > DBL_EPSILON;
        case expr::not_ee: return (abs(evaluate_const_expr(ex->left)) <= DBL_EPSILON); //if left is equal zero, return 1 else 0
        case expr::conditional_ee: return abs(evaluate_const_expr(ex->left)) > DBL_EPSILON //if left is not zero, go into cond otherwise goto else
            ? evaluate_const_expr(ex->right)
            : evaluate_const_expr(ex->conditional_else);
        case expr::compiled_ee: throw std::runtime_error("Tried evaluating compiled expression as constant expression in optimizer.");
        case expr::pn_compiled_ee: 
        case expr::pn_skips_ee: throw std::runtime_error("Tried evaluating pn compiled expression as constant expression in optimizer.");
        default: throw std::runtime_error("unknown operand found in const expression evaluation");
    }
}

inline arr<constraint> to_constraint_array(const std::list<constraint*>& con_lst)
{
    const int size = static_cast<int>(con_lst.size());
    constraint* store = static_cast<constraint*>(malloc(sizeof(constraint)*size));
    int i = 0;
    for (const constraint* con : con_lst)
    {
        store[i++] = *con;
    }
    return { store, size };
}

inline void create_false_constraint_set(arr<constraint>* con_array)
{
    constraint* con = new constraint();
    
    expr* left = new expr();
    left->operand = expr::literal_ee;
    left->left = nullptr;
    left->right = nullptr;
    left->value = 1.0;

    expr* right = new expr();
    left->operand = expr::literal_ee;
    left->left = nullptr;
    left->right = nullptr;
    left->value = 0.0;

    con->uses_variable = false;
    con->operand = constraint::less_c;
    con->value = left;
    con->expression = right;
    *con_array = {con, 1}; 
}

void domain_optimization_visitor::reduce_constraint_set(arr<constraint>* con_array)
{
    std::list<constraint*> temp_lst;
    for (int i = 0; i < con_array->size; ++i)
    {
        constraint* temp = &con_array->store[i];
        if(temp->uses_variable || !is_const_constraint(temp))
        {
            temp_lst.push_back(temp);
            continue;
        }

        const bool valid = evaluate_const_constraint(temp);
        if(!valid)
        {
            create_false_constraint_set(con_array); 
            return;
        }
    }

    if(static_cast<int>(temp_lst.size()) < con_array->size)
    {
        *con_array = to_constraint_array(temp_lst);
    }
}

void domain_optimization_visitor::reduce_expr(expr* ex)
{
    if(ex->operand == expr::literal_ee) return; //cannot reduce further
    if(!is_const_expr(ex)) return;

    ex->value = evaluate_const_expr(ex); //this has to occur before we change the modify the expr.
    ex->operand = expr::literal_ee;
    ex->left = nullptr;
    ex->right = nullptr;
}
