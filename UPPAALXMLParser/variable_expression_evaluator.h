
#ifndef UPDATE_EXPRESSION_EVALUATOR_H
#define UPDATE_EXPRESSION_EVALUATOR_H

#include "../engine/Domain.h"
#include <unordered_map>

#define PI 3.14159265358979323846 

class expression;
using namespace std;
 
enum update_types { UPDATEDELIMITER = 1, UPDATEVARIABLE, UPDATENUMBER, UPDATEFUNCTION };
const int UPDATE_NUMVARS = 26;
class variable_expression_evaluator {
    char *exp_ptr_; // points to the expression
    char token_[256]; // holds current token
    char tok_type_; // holds token's type
    unordered_map<string, double>* const_local_vars_;
    unordered_map<string, double>* const_global_vars_;
    unordered_map<string, int>* local_vars_;
    unordered_map<string, int>* global_vars_;
    expr* eval_exp1();
    expr* eval_exp2();
    expr* eval_exp3();
    expr* eval_exp4();
    expr* eval_exp5();
    expr* eval_exp6();
    expr* eval_exp(char *exp);
    void get_token();
public:
    variable_expression_evaluator(
        unordered_map<string,int>* local_vars, unordered_map<string,int>* global_vars,
        unordered_map<string, double>* const_local_vars,unordered_map<string, double>* const_global_vars);
    static expr* evaluate_variable_expression(
        const string& input, unordered_map<string, int>* local_vars, unordered_map<string, int>* global_vars,
        unordered_map<string, double>* const_local_vars,unordered_map<string, double>* const_global_vars);
    
    char errormsg[64];
};


#endif