    #ifndef DOMAIN_H
#define DOMAIN_H

struct state;
struct edge;
struct node;
int min(int, int);

#include "../common/macro.h"
#include "../common/my_stack.h"

#define HAS_HIT_MAX_STEPS(x) ((x) < 0)

template<typename T>
struct arr
{
    T* store;
    int size;

    static arr<T> empty(){ return arr<T>{nullptr, 0}; }
};

#define IS_LEAF(x) ((x) < 2)
struct expr  // NOLINT(cppcoreguidelines-pro-type-member-init)
{
    enum operators
    {
        //value types
        literal_ee = 0,
        clock_variable_ee = 1,

        //random
        random_ee,

        //arithmatic types
        plus_ee,
        minus_ee,
        multiply_ee,
        division_ee,
        power_ee,
        negation_ee,
        sqrt_ee,
        modulo_ee,

        //boolean types
        and_ee,
        or_ee,
        less_equal_ee,
        greater_equal_ee,
        less_ee,
        greater_ee,
        equal_ee,
        not_equal_ee,
        not_ee,

        //conditional types
        conditional_ee,
        compiled_ee,
        pn_compiled_ee,
        pn_skips_ee,
        
    } operand = literal_ee;
    
    expr* left = nullptr;
    expr* right = nullptr;

    union
    {
        double value = 1.0;
        int variable_id;
        int length;
        expr* conditional_else;
        int compile_id;
    };

    CPU GPU double evaluate_expression(state* state);
};


/**
 * \brief Takes in constraint::operators and returns bool whether the operand is a constraint
 * \param a constraint::operators
 */
#define IS_INVARIANT(a) ((a) < 2)
struct constraint
{
    enum operators
    {
        less_equal_c = 0,
        less_c = 1,
        greater_equal_c = 2,
        greater_c = 3,
        equal_c = 4,
        not_equal_c = 5,
        compiled_c
    } operand;

    bool uses_variable;
    union //left hand side
    {
        expr* value;
        int variable_id;
        int compile_id;
    };
    expr* expression; //right hand side
    CPU GPU bool evaluate_constraint(state* state) const;
    CPU GPU static bool evaluate_constraint_set(const arr<constraint>& con_arr, state* state);
};

struct clock_var
{
    int id;
    bool should_track;
    unsigned rate;
    double value;
    double max_value;

    CPU GPU void add_time(const double time);
    CPU GPU void set_value(const double val);
};


#define IS_URGENT(x) ((x) > 2)
struct node
{
    int id{};
    enum node_types
    {
        location = 0,
        goal = 1,
        branch = 2,
        urgent = 3,
        committed = 4,
    } type = location;
    expr* lamda{};
    arr<edge> edges = arr<edge>::empty();
    arr<constraint> invariants = arr<constraint>::empty();
    CPU GPU double max_progression(state* state, bool* is_finite) const;
};

struct update
{
    int variable_id;
    expr* expression;
    CPU GPU void apply_update(state* state) const;
};


#define TAU_CHANNEL 0
#define IS_TAU(x) ((x) == 0)
#define IS_LISTENER(x) ((x) < 0)
#define CAN_SYNC(brod, list) ((brod) == (-(list)))
#define IS_BROADCASTER(x) ((x) > 0)


struct edge
{
    int channel{};
    expr* weight{};
    node* dest{};
    arr<constraint> guards = arr<constraint>::empty();
    arr<update> updates = arr<update>::empty();
    CPU GPU void apply_updates(state* state) const;
    CPU GPU bool edge_enabled(state* state) const;
};

// struct proposition
// {
//     enum prop_type
//     {
//         reach,
//         sys_constraint,s
//     } type;
//     union
//     {
//         struct
//         {
//             int process_id;
//             int id;
//         } reachability;
//         constraint constraint;
//     } data;
//
//     bool evaluate(state* state) const;
// };
//
// #define QUERY_GOAL (-1)
// #define QUERY_TERMINAL (-2)
// #define IS_TERMINAL(x) ((x) == -2)
// #define IS_GOAL(x) ((x)==-1)
// struct query
// {
//     enum query_types
//     {
//         liveness,
//         safety,
//         estimate
//     } type;
//     const int inputs;
//     const int states;
//     int* dfa;
//     arr<proposition> propositions;
//
// #define SET_BIT(i, x) ((x) | (1<<(i)))
// #define SET_BIT_IF(i, cond, x) ((x) | ((cond)<<(i)))
// #define IS_BIT_SET(i, x) ((x) & (1<<(i))) 
//     CPU GPU bool check_query(state* state) const;
// };

struct network
{
    arr<node*> automatas;
    arr<clock_var> variables;
    // query* query;
};


struct state
{
    int query_state;
    unsigned urgent_count;
    unsigned committed_count;
    unsigned simulation_id;
    unsigned steps;
    double global_time;

    arr<node*> models;
    arr<clock_var> variables;

    struct w_edge
    {
        edge* e;
        double w;
    };
    
    curandState* random;
    my_stack<expr*> expr_stack;
    my_stack<double> value_stack;
    my_stack<w_edge> edge_stack;

    CPU GPU void traverse_edge(int process_id, node* dest);
    CPU GPU void broadcast_channel(const int channel, const int process);
    CPU GPU static state init(void* cache, curandState* random, const network* model, const unsigned expr_depth, const unsigned backtrace_depth, const
                              unsigned fanout);
    CPU GPU void reset(const unsigned sim_id, const network* model, const unsigned initial_urgent_count, const unsigned
                       initial_committed_count);
};
#endif
