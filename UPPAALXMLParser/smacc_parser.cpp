#include "smacc_parser.h"

#include <iostream>

#include "variable_expression_evaluator.h"
#define THROW_LINE(arg); throw parser_exception(arg, __FILE__, __LINE__);

template<typename T>
arr<T> to_array(std::list<T>* list)
{
    int size = static_cast<int>(list->size());
    if(size == 0) return arr<T>::empty();
    T* array = static_cast<T*>(malloc(sizeof(T)*size));

    int i = 0;
    for(T& item : *list)
    {
        array[i] = item;
        i++;
    }

    return arr<T>{array, size};
}

int smacc_parser::get_variable_id(const std::string& varname) const
{
    if (this->global_variables_.count(varname))
    {
        return this->global_variables_.at(varname);
    }
    else if (
        this->local_variables_.count(this->current_process_) &&
        this->local_variables_.at(this->current_process_).count(varname))
    {
        return this->local_variables_
            .at(this->current_process_)
            .at(varname);
    }
    else
        throw std::runtime_error("Variable '" + varname + "' could not be found in global or local scope");
}

void smacc_parser::parse_automata(const pugi::xml_node& root)
{
    //order MUST be vars, nodes, edges.
    //the tree uses this order to not miss out on any information
    for(pugi::xml_node& var : root.children("Variable"))
    {
        this->parse_variable(var, true);
    }
    
    for(pugi::xml_node& node : root.children("Node"))
    {
        this->parse_node(node);
    }

    for(pugi::xml_node& edge : root.children("Edge"))
    {
        this->parse_edge(edge);
    }
}

void smacc_parser::parse_node(const pugi::xml_node& root)
{
    const std::string true_str = "True";

    node* n = new node();
    n->id = root.attribute("id").as_int();
    n->edges = arr<edge>::empty();
    n->type = static_cast<node::node_types>(root.attribute("type").as_int(node::location));
    const bool init = root.attribute("init").as_bool();
    
    std::string name = root.attribute("name").as_string();
    const std::string lambda = root.attribute("lambda").as_string();
    std::list<constraint> inv_lst{};

    n->lamda = this->parse_expr_from_string(lambda);
    
    for(pugi::xml_node& invariant : root.children("Constraints"))
    {
        this->parse_constraint(invariant, inv_lst);
    }

    n->invariants = to_array<constraint>(&inv_lst);

    if (init)
    {
        this->start_nodes_.push_back(n);
    }
    this->system_map_.insert(std::pair<int, int>(n->id, this->current_process_));
    this->node_names_.insert(std::pair<int, std::string>(n->id, name));
    this->node_map_.insert(std::pair<int, node*>(n->id, n));
}

inline bool smacc_parser::check_is_clock(const int var_id) const
{
    if(this->var_map_.count(var_id))
        return this->var_map_.at(var_id).rate > 0;

    throw std::runtime_error("Could not find clock with variable");
}

void smacc_parser::parse_constraint(const pugi::xml_node& root, std::list<constraint>& lst)
{
    constraint con{};
    bool flip_operator = false;
    const auto left_var = root.attribute("left_var");
    const auto right_var = root.attribute("right_var");
    
    if (left_var && right_var)
    {
        const int v_id_1 = get_variable_id(left_var.as_string());
        const int v_id_2 = get_variable_id(right_var.as_string());
        const bool left_clock = this->check_is_clock(v_id_1);
        const bool right_clock = this->check_is_clock(v_id_2);
        
        if(left_clock && right_clock)
            throw std::runtime_error("Constraint with clocks on both sides detected.");
        if ((left_clock && !right_clock) || (!left_clock && !right_clock))
        {
            con.variable_id = v_id_1;
            con.expression = new expr{ expr::clock_variable_ee, nullptr, nullptr };
            con.expression->variable_id = v_id_2;
        }
        else if (!left_clock && right_clock)
        {
            con.variable_id = v_id_2;
            con.expression = new expr{ expr::clock_variable_ee, nullptr, nullptr };
            con.expression->variable_id = v_id_1;
        }
        // else this is handled through first case
        // {
        //     con.variable_id = v_id_1;
        //     con.expression = new expr{ expr::clock_variable_ee, nullptr, nullptr };
        //     con.expression->variable_id = v_id_2;
        // }
    }
    else if (!left_var && right_var) //left expression, right variable
    {
        flip_operator = true;
        const std::string rn = right_var.as_string();
        con.uses_variable = true;
        con.variable_id = get_variable_id(rn);
        con.expression = parse_expr_from_string( root.attribute("left_expr").as_string());
    }
    else if (left_var && !right_var)
    {
        flip_operator = false;
        const std::string ln = left_var.as_string();
        con.uses_variable = true;
        con.variable_id = get_variable_id(ln);
        con.expression = parse_expr_from_string( root.attribute("right_expr").as_string() );
    }
    else if (!left_var && !right_var)
    {
        flip_operator = false;
        con.uses_variable = false;
        con.value = parse_expr_from_string(
            root.attribute("left_expr").as_string() );
        con.expression = parse_expr_from_string(
            root.attribute("right_expr").as_string() );
    }

    const std::string op = root.attribute("type").as_string(); 
    
    if(op == "<") con.operand = constraint::less_c;
    else if (op == "<=") con.operand = constraint::less_equal_c;
    else if (op == ">") con.operand = constraint::greater_c;
    else if (op == ">=") con.operand = constraint::greater_equal_c;
    else if (op == "==") con.operand = constraint::equal_c;
    else if (op == "!=") con.operand = constraint::not_equal_c;
    else throw std::runtime_error("Constraint type not recognized");

    if (flip_operator)
    {
        if(con.operand == constraint::greater_c) con.operand = constraint::less_c;
        else if(con.operand == constraint::greater_equal_c) con.operand = constraint::less_equal_c;
        else if(con.operand == constraint::less_c) con.operand = constraint::greater_c;
        else if(con.operand == constraint::less_equal_c) con.operand = constraint::greater_equal_c;
    }

    lst.push_back(con);
}

void smacc_parser::parse_edge(const pugi::xml_node& root)
{
    edge e{};

    const int local_source = root.attribute("source_id").as_int();
    const int local_dest = root.attribute("dest_id").as_int();
    std::list<update> update_lst{};
    std::list<constraint> con_lst{};
    
    e.dest = this->node_map_.at(local_dest);
    e.channel = root.attribute("channel").as_int();
    e.weight = parse_expr_from_string(root.attribute("weight").as_string());

    
    for(pugi::xml_node& update : root.children("Update"))
    {
        this->parse_update(update, update_lst);
    }

    for(pugi::xml_node& constraint : root.children("Constraint"))
    {
        this->parse_constraint(constraint, con_lst);
    }

    e.updates = to_array<update>(&update_lst);
    e.guards = to_array<constraint>(&con_lst);

    if(this->outgoing_edge_map_.count(local_source) == 0)
        this->outgoing_edge_map_.insert( std::pair<int, std::list<edge>>(local_source, std::list<edge>()) );
    this->outgoing_edge_map_.at(local_source).push_back(e);
}

void smacc_parser::parse_variable(const pugi::xml_node& root, const bool is_local)
{
    clock_var var{};

    std::string name = root.attribute("name").as_string();
    var.id = this->variable_count_++;
    var.value = root.attribute("value").as_double();
    var.rate = root.attribute("rate").as_int();
    var.should_track = root.attribute("track").as_bool();
    var.max_value = var.value;
    
    if(is_local)
    {
        std::string contextualised_name = name + "(p" + std::to_string(this->current_process_) + ")";
        this->clock_names_.insert(std::pair<int, std::string>(var.id, contextualised_name));

        if(!this->local_variables_.count(this->current_process_))
            this->local_variables_.insert(
                std::pair<int, unordered_map<std::string, int >>
                (this->current_process_, unordered_map<std::string, int>()));
        
        this->local_variables_
            .at(this->current_process_)
            .insert(std::pair<std::string, int>(name, var.id));
    }
    else
    {
        this->clock_names_.insert(std::pair<int, std::string>(var.id, name));
        this->global_variables_.insert(std::pair<std::string, int>(name, var.id));
    }

    this->var_map_.insert(std::pair<int, clock_var>(var.id, var));
}

void smacc_parser::parse_update(const pugi::xml_node& root, std::list<update>& update_lst)
{
    printf("hello update!\n");  
    update up{};
    up.variable_id = get_variable_id(root.attribute("var").as_string());
    up.expression = this->parse_expr_from_string(root.attribute("expr").as_string());
    update_lst.push_back(up);
}

void smacc_parser::post_process_nodes()
{
    for(auto& pair : this->node_map_)
        if(this->outgoing_edge_map_.count(pair.first))
            pair.second->edges = to_array<edge>(&this->outgoing_edge_map_.at(pair.first));
}


expr* smacc_parser::parse_expr_from_string(const std::string& expr)
{
    unordered_map<std::string, double> const_empty_map{}; //needs this as dummy. Should be empty.
    unordered_map<std::string, int>* global_vars = &this->global_variables_;
    unordered_map<std::string, int>* local_vars =  (this->local_variables_.count(this->current_process_)
        ? &(this->local_variables_.at(this->current_process_))
        : &this->global_variables_);
    
    //global variables are used to avoid additional allocation.
    //If local variables are used as a secondary priority,
    //so if it misses global vars first, it will miss again, ensuring proper error msg
    
    return variable_expression_evaluator::evaluate_variable_expression(expr,
                                                                       local_vars,
                                                                       global_vars,
                                                                       &const_empty_map,
                                                                       &const_empty_map);
}

arr<clock_var> smacc_parser::to_var_array() const
{
    const int size = static_cast<int>(this->var_map_.size());
    clock_var* store = static_cast<clock_var*>(malloc(sizeof(clock_var)*size));

    for(const auto& pair : this->var_map_)
    {
        store[pair.first] = pair.second;
    }
    return arr<clock_var>{ store, size };
}

network smacc_parser::parse(const std::string& file)
{
    pugi::xml_document doc;

    if(!doc.load_file(file.c_str()))
    {
        THROW_LINE("FILE NOT FOUND")
    }

    this->variable_count_ = 0; //ensure starting from 0
    this->current_process_ = 0; //ensure starting from 0
    for(pugi::xml_node& var : doc.child("Network").children("Variable"))
    {
        this->parse_variable(var, false);
    }

    for(pugi::xml_node& automata : doc.child("Network").children("Automata"))
    {
        this->parse_automata(automata);
        this->current_process_++;
    }

    this->post_process_nodes();

    network n{};
    n.automatas = to_array<node*>(&this->start_nodes_);
    n.variables = to_var_array();

    return n;
}
