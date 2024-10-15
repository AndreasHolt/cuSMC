#ifndef SMACC_PARSER_H
#define SMACC_PARSER_H

#include <unordered_map>
#include "../engine/Domain.h"
#include "pugixml.hpp"
#include "abstract_parser.h"


class smacc_parser final : public abstract_parser
{
private:
    int current_process_ = 0;
    int variable_count_ = 0;
    std::unordered_map<std::string, int> global_variables_;
    std::unordered_map<int, std::unordered_map<std::string, int>> local_variables_;
    std::unordered_map<int, clock_var> var_map_;
    std::unordered_map<int, node*> node_map_;
    std::list<node*> start_nodes_;
    std::unordered_map<int, std::string> node_names_;
    std::unordered_map<int, std::string> clock_names_;
    std::unordered_map<int, int> system_map_;
    std::unordered_map<int, std::list<edge>> outgoing_edge_map_; //node_id to list of edges

    int get_variable_id(const std::string& varname) const;
    
    void parse_automata(const pugi::xml_node& root);
    void parse_node(const pugi::xml_node& root);
    bool check_is_clock(int var_id) const;
    void parse_constraint(const pugi::xml_node& root, std::list<constraint>& lst);
    void parse_edge(const pugi::xml_node& root);
    void parse_variable(const pugi::xml_node& root, const bool is_local = false);
    void parse_update(const pugi::xml_node& root, std::list<update>&);
    void post_process_nodes();

    expr* parse_expr_from_string(const std::string& expr);
    arr<clock_var> to_var_array() const;
public:
    network parse(const std::string& file) override;
    std::unordered_map<int, std::string>* get_nodes_with_name() override { return &this->node_names_; }
    std::unordered_map<int, int>* get_subsystems() override { return &this->system_map_; }
    std::unordered_map<int, std::string>* get_clock_names() override{ return &this->clock_names_; }
    std::unordered_map<int, std::string>* get_template_names() override {return new std::unordered_map<int, std::string>();}
};

#endif