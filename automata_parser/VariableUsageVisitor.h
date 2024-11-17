//
// Created by andwh on 08/11/2024.
//

#ifndef VARIABLEUSAGEVISITOR_H
#define VARIABLEUSAGEVISITOR_H
#pragma once


#include "../include/VariableTypes.h"
#include <unordered_map>

#include "../engine/Domain.h"
#include "../network_optimization/visitor.h"
#include <unordered_set>




class VariableTrackingVisitor : public visitor {
public:
    struct VariableUsage {
        std::string name;
        VariableKind kind;
        std::string template_name;  // Which template this variable belongs to ("" for global)
        bool is_const;
        std::unordered_set<int> used_in_nodes;  // Changed from vector to set
        std::unordered_set<int> used_in_edges;  // Changed from vector to set
        int rate;

        // Add default constructor
        VariableUsage() :
            name(""),
            kind(VariableKind::INT),
            template_name(""),
            is_const(false),
            rate(0) {}

        // Keep existing constructor
        VariableUsage(const std::string& n, VariableKind k,
                     const std::string& t = "", bool c = false, int r = 0) :
            name(n),
            kind(k),
            template_name(t),
            is_const(c),
            rate(r) {}
    };



private:
    std::unordered_map<int, VariableUsage> variable_registry_;
    std::unordered_map<std::string, int> scope_to_id_map_;
    std::string current_template_;  // Track current template being visited
    int current_node_id_ = -1;
    int current_edge_id_ = -1;
    int next_var_id_ = 0;

    void track_expr_variables(expr* e) {
        if(e == nullptr) return;

        if(e->operand == expr::clock_variable_ee) {
            auto it = variable_registry_.find(e->variable_id);
            if(it != variable_registry_.end()) {
                if(current_node_id_ >= 0) {
                    it->second.used_in_nodes.insert(current_node_id_);  // Changed from push_back to insert
                }
                if(current_edge_id_ >= 0) {
                    it->second.used_in_edges.insert(current_edge_id_);  // Changed from push_back to insert
                }
            }
        }

        track_expr_variables(e->left);
        track_expr_variables(e->right);
        if(e->operand == expr::conditional_ee) {
            track_expr_variables(e->conditional_else);
        }
    }


    // Helper function to create unique scope key
    std::string make_scope_key(const std::string& var_name, const std::string& template_name = "") {
        return template_name.empty() ? "global:" + var_name : template_name + ":" + var_name;
    }

    // Helper function to register a variable
    int register_variable(const std::string& var_name, VariableKind kind,
                     const std::string& template_name, bool is_const, int rate) {
        std::string scope_key = make_scope_key(var_name, template_name);

        printf("Attempting to register variable '%s' in scope '%s'\n",
               var_name.c_str(), scope_key.c_str());

        // Check if variable already exists in this scope
        auto it = scope_to_id_map_.find(scope_key);
        if(it != scope_to_id_map_.end()) {
            printf("Variable already exists with ID %d\n", it->second);
            return it->second;
        }

        int var_id = next_var_id_++;
        scope_to_id_map_[scope_key] = var_id;

        variable_registry_[var_id] = VariableUsage(
            var_name,
            kind,
            template_name,
            is_const,
            rate
        );

        printf("Registered %s variable '%s' (id=%d) in %s with rate=%d\n",
               kind == VariableKind::CLOCK ? " clock" :
               kind == VariableKind::INT ? "local int" : "global int",
               var_name.c_str(),
               var_id,
               template_name.empty() ? "global scope" : template_name.c_str(),
               rate);

        return var_id;
    }



public:
    // New method to handle template-local variables
    void register_template_variables(const std::string& template_name,
                                   const std::vector<clock_var>& vars) {
        current_template_ = template_name;

        for(const auto& var : vars) {
            // std::string var_name = get_var_name_from_parser(var.id);
            std::string var_name = "test";

            register_variable(
                var_name,
                var.rate > 0 ? VariableKind::CLOCK : VariableKind::INT,
                template_name,
                false,
                var.rate
            );
        }
    }


    void visit(network* net) override {
        if(has_visited(net)) return;

        current_template_ = "";  // Ensure we're in global scope

        // Debug print network info
        printf("\nProcessing network with %d variables\n", net->variables.size);

        // Process global variables
        for(int i = 0; i < net->variables.size; i++) {
            const clock_var& var = net->variables.store[i];

            printf("Processing variable %d: id=%d, rate=%d\n",
                   i, var.id, var.rate);

            // Temporary name until we implement proper name lookup
            std::string var_name = "var_" + std::to_string(var.id);

            int var_id = register_variable(
                var_name,
                var.rate > 0 ? VariableKind::CLOCK : VariableKind::INT,
                "",  // global scope
                false,  // not const
                var.rate
            );

            // Debug print registration result
            printf("Registered as variable ID %d\n", var_id);
        }

        printf("\nFinished processing network variables\n");

        accept(net, this);
    }


    void visit(node* n) override {
        if(has_visited(n)) return;

        current_node_id_ = n->id;
        printf("Visiting node %d\n", n->id);

        // Process invariants
        for(int i = 0; i < n->invariants.size; i++) {
            const constraint& inv = n->invariants.store[i];

            if(inv.uses_variable) {
                auto it = variable_registry_.find(inv.variable_id);
                if(it != variable_registry_.end()) {
                    it->second.used_in_nodes.insert(n->id);  // Changed from push_back
                }
            }

            // Track variables in expressions
            track_expr_variables(inv.expression);
            if(!inv.uses_variable) {
                track_expr_variables(inv.value);
            }
        }

        accept(n, this);
    }


    void visit(edge* e) override {
        if(has_visited(e)) return;

        current_edge_id_++;
        printf("Visiting edge %d\n", current_edge_id_);

        // Process guards
        for(int i = 0; i < e->guards.size; i++) {
            const constraint& guard = e->guards.store[i];

            if(guard.uses_variable) {
                auto it = variable_registry_.find(guard.variable_id);
                if(it != variable_registry_.end()) {
                    it->second.used_in_edges.insert(current_edge_id_);  // Changed from push_back
                }
            }

            track_expr_variables(guard.expression);
            if(!guard.uses_variable) {
                track_expr_variables(guard.value);
            }
        }

        // Process updates
        for(int i = 0; i < e->updates.size; i++) {
            const update& upd = e->updates.store[i];

            auto it = variable_registry_.find(upd.variable_id);
            if(it != variable_registry_.end()) {
                it->second.used_in_edges.insert(current_edge_id_);  // Changed from push_back
            }

            track_expr_variables(upd.expression);
        }

        accept(e, this);
    }


    // Required implementations of pure virtual functions
    void visit(constraint* c) override {
        if(has_visited(c)) return;
        if(c->uses_variable) {
            auto it = variable_registry_.find(c->variable_id);
            if(it != variable_registry_.end()) {
                if(current_node_id_ >= 0) {
                    it->second.used_in_nodes.insert(current_node_id_);  // Changed from push_back
                }
                if(current_edge_id_ >= 0) {
                    it->second.used_in_edges.insert(current_edge_id_);  // Changed from push_back
                }
            }
        }
        track_expr_variables(c->expression);
        accept(c, this);
    }


    void visit(clock_var* cv) override {
        if(has_visited(cv)) return;
        // Add tracking if needed
        accept(cv, this);
    }

    void visit(update* u) override {
        if(has_visited(u)) return;
        auto it = variable_registry_.find(u->variable_id);
        if(it != variable_registry_.end()) {
            if(current_edge_id_ >= 0) {
                it->second.used_in_edges.insert(current_edge_id_);  // Changed from push_back
            }
        }
        track_expr_variables(u->expression);
        accept(u, this);
    }


    void visit(expr* e) override {
        if(has_visited(e)) return;
        track_expr_variables(e);
        accept(e, this);
    }

    void clear() override {
        visitor::clear();
        variable_registry_.clear();
        scope_to_id_map_.clear();
        current_template_ = "";
        current_node_id_ = -1;
        current_edge_id_ = -1;
        next_var_id_ = 0;
    }

    // Public interface to get results
    const std::unordered_map<int, VariableUsage>& get_variable_registry() const {
        return variable_registry_;
    }

    int get_variable_id(const std::string& scope_key) const {
        auto it = scope_to_id_map_.find(scope_key);
        return it != scope_to_id_map_.end() ? it->second : -1;
    }

    void print_variable_usage() const {
        printf("\nVariable Usage Report:\n");
        printf("=====================\n");

        // Print globals first
        printf("\nGlobal Variables:\n");
        for(const auto& pair : variable_registry_) {
            const auto& var = pair.second;
            if(var.template_name.empty()) {
                print_variable(pair.first, var);
            }

        }

        // Print locals grouped by template
        std::unordered_map<std::string, std::vector<std::pair<int, const VariableUsage*>>> template_vars;
        for(const auto& pair : variable_registry_) {
            const auto& var = pair.second;
            if(!var.template_name.empty()) {
                template_vars[var.template_name].push_back({pair.first, &var});
            }
        }

        for(const auto& template_pair : template_vars) {
            printf("\nTemplate %s Variables:\n", template_pair.first.c_str());
            for(const auto& var_pair : template_pair.second) {
                print_variable(var_pair.first, *var_pair.second);
            }
        }

        printf("=====================\n");
    }

private:
    void print_variable(int id, const VariableUsage& var) const {
        printf("\n  Variable '%s' (ID %d):\n", var.name.c_str(), id);
        printf("    Kind: %s\n",
               var.kind == VariableKind::INT ? "INT" :
               var.kind == VariableKind::CLOCK ? "CLOCK" : "CLOCK");
        if(!var.template_name.empty()) {
            printf("    Template: %s\n", var.template_name.c_str());
        }
        printf("    Rate: %d\n", var.rate);
        printf("    Used in nodes: ");
        for(int node : var.used_in_nodes) printf("%d ", node);
        printf("\n    Used in edges: ");
        for(int edge : var.used_in_edges) printf("%d ", edge);
        printf("\n");
    }


};



#endif //VARIABLEUSAGEVISITOR_H
