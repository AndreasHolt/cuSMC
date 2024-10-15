#pragma once
#ifndef INTERPRETR_H
#define INTERPRETR_H

#include <curand_kernel.h>
#include <curand_uniform.h>

#include "../common/my_stack.h"
#include "Domain.h"

union i_val
{
    typedef signed long int int_t;
    typedef double float_t;

    i_val() = default;

    i_val(const float_t fp)
    {
        this->fp = fp;
    }

    i_val(const int i)
    {
        this->i = i;
    }

    i_val(const int_t i)
    {
        this->i = i;
    }

    long int i{};
    double fp;
};

struct automata
{
    node* initial_node;
    arr<i_val> local_variables;
};

struct sim_state
{
    typedef double clock_t;
    struct edge_w
    {
        edge* e;
        double w;
    };

    struct automata_state
    {
        node* current;
        arr<i_val> local_variables;
    };

    clock_t* clocks;
    int* clock_rates;

    my_stack<i_val> intp_stack;
    my_stack<edge_w> edge_stack;
    curandState* random;
    
    i_val* global_variables;
    int current_process;
    automata_state* process_states;

    void update_rate(int clock_id);
    void reset_rate(int clock_id);
};

struct pn_expr
{
    #define USE_1POP(x) ((x) | (1 << 29)) 
    #define USE_2POP(x) ((x) | (1 << 30)) 
    #define USES_1POP(x) ((x) & (1<<29))
    #define USES_2OP(x) ((x) & (1<<30))
    enum types
    {
        //axioms
        literal_s = 0, 
        variable_s = 1,
        local_variable_s = 2,
        random_s = 3,

        //control
        cgoto_s = 4,

        //conversions
        i_fp_conversion,
        fp_i_conversion,
        
        //int artihmatic
        i_plus_s, 
        i_minus_s, 
        i_multiply_s,
        modulo_s,
        
        //float arithmatic
        fp_plus_s, 
        fp_minus_s, 
        fp_multiply_s,
        
        pow_s,
        sqrt_s,
        ln_s,

        //boolean
        negation_s,
        less_s,
        less_equal_s,
        greater_s,
        greater_equal_s,
        equal_s,
        not_equal_s,
        and_s,
        or_s,

        //compiled,
        compiled_s
    } type;

    union expr_metadata
    {
        i_val value;
        int variable_id{};
        int goto_dest;
    } data{};

    i_val evaluate(sim_state* state) const;
};

inline i_val pn_expr::evaluate(sim_state* state) const
{
    constexpr i_val::int_t neg_flip = static_cast<i_val::int_t>(1U << (sizeof(i_val::int_t)-1)); 
    constexpr i_val def = 0.0;
    i_val v1;
    i_val v2;
    switch(this->type)
    {
    case literal_s: return this->data.value;
    case variable_s: return state->global_variables[this->data.variable_id];
    case local_variable_s:
        return state->process_states[state->current_process].local_variables[this->data.variable_id];
    case random_s: return curand_uniform_double(state->random);
    case cgoto_s: return def;
    case i_fp_conversion:
        v1 = state->intp_stack.pop();
        return static_cast<i_val::float_t>(v1.i);
    case fp_i_conversion:
        v1 = state->intp_stack.pop();
        return static_cast<i_val::int_t>(v1.fp);
    case i_plus_s:
        v2 = state->intp_stack.pop();
        v1 = state->intp_stack.pop();
        return v1.i + v2.i;
    case i_minus_s:
        v2 = state->intp_stack.pop();
        v1 = state->intp_stack.pop();
        return v1.i - v2.i;
    case i_multiply_s:
        v2 = state->intp_stack.pop();
        v1 = state->intp_stack.pop();
        return v1.i * v2.i;
    case modulo_s:
        v2 = state->intp_stack.pop();
        v1 = state->intp_stack.pop();
        return (v1.i % v2.i);
    case fp_plus_s:
        v2 = state->intp_stack.pop();
        v1 = state->intp_stack.pop();
        return v1.fp + v2.fp;
    case fp_minus_s:
        v2 = state->intp_stack.pop();
        v1 = state->intp_stack.pop();
        return v1.fp - v2.fp;
    case fp_multiply_s:
        v2 = state->intp_stack.pop();
        v1 = state->intp_stack.pop();
        return v1.fp * v2.fp;
    case pow_s:
        v2 = state->intp_stack.pop();
        v1 = state->intp_stack.pop();
        return pow(v1.fp, v2.fp);
    case sqrt_s:
        v1 = state->intp_stack.pop();
        return sqrt(v1.fp);
    case ln_s:
        v1 = state->intp_stack.pop();
        return log(v1.fp);
    case negation_s:
        v1 = state->intp_stack.pop();
        return neg_flip ^ v1.i; //works for both float and ints
    case less_s:
        v1 = state->intp_stack.pop();
        v2 = state->intp_stack.pop();
        return v1.i < v2.i;
    case less_equal_s:
        v1 = state->intp_stack.pop();
        v2 = state->intp_stack.pop();
        return v1.i <= v2.i;
    case greater_s:
        v1 = state->intp_stack.pop();
        v2 = state->intp_stack.pop();
        return v1.i > v2.i;
    case greater_equal_s:
        v1 = state->intp_stack.pop();
        v2 = state->intp_stack.pop();
        return v1.i <= v2.i;
    case equal_s:
        v1 = state->intp_stack.pop();
        v2 = state->intp_stack.pop();
        return v1.i == v2.i;
    case not_equal_s:
        v1 = state->intp_stack.pop();
        v2 = state->intp_stack.pop();
        return v1.i != v2.i;
    case and_s:
        v1 = state->intp_stack.pop();
        v2 = state->intp_stack.pop();
        return v1.i && v2.i;
    case or_s:
        v1 = state->intp_stack.pop();
        v2 = state->intp_stack.pop();
        return v1.i || v2.i;
    case compiled_s:
        return def;
    }
    return def;
}

struct function
{
    pn_expr* statements;
    int size;

    i_val call(sim_state* state) const
    {
        state->intp_stack.clear();
        for (int i = 0; i < size; ++i)
        {
            const pn_expr* stmt = &statements[i];
            
            if(stmt->type == pn_expr::cgoto_s) //TODO this can be moved to eval scope
                i = i + (stmt->data.goto_dest * (state->intp_stack.pop().i != 0));
            else
                state->intp_stack.push_val(stmt->evaluate(state));
        }
        return state->intp_stack.pop();
    }
};

#endif

