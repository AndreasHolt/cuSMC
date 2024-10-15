#include "variable_expression_evaluator.h"
#include "helper_methods.h"

// Parser constructor.
variable_expression_evaluator::variable_expression_evaluator(
    unordered_map<string,int>* local_vars, unordered_map<string,int>* global_vars,
    unordered_map<string, double>* const_local_vars,unordered_map<string, double>* const_global_vars)
{
    exp_ptr_ = nullptr;
    const_local_vars_ = const_local_vars;
    const_global_vars_ = const_global_vars; 
    local_vars_ = local_vars;
    global_vars_ = global_vars;
}

// Parser entry point.
expr* variable_expression_evaluator::eval_exp(char *exp)
{
    
    exp_ptr_ = exp;
    get_token();
    
    if (!*token_) 
    {
        THROW_LINE("No Expression Present")
    }
    
    expr* result = eval_exp1();
    if (*token_) // last token must be null
    {
        THROW_LINE("Syntax Error")
    }
    return result;
}
// Process an assignment.
expr* variable_expression_evaluator::eval_exp1()
{
    int slot;
    expr* result;
    if (tok_type_ == UPDATEVARIABLE) 
    {
        char temp_token[80];
        // save old token
        char *t_ptr = exp_ptr_;
        strcpy(temp_token, token_);
        // compute the index of the variable
        slot = *token_ - 'A';
        get_token();
        if (*token_ != '=') 
        {
            exp_ptr_ = t_ptr; // return current token
            strcpy(token_, temp_token); // restore old token
            tok_type_ = UPDATEVARIABLE;
        }
        else {
            get_token(); // get next part of exp
            
            result = eval_exp2();
            //vars[slot] = result;
            return result;
        }
    }
    result = eval_exp2();
    return result;
}
// Add or subtract two terms.
 expr* variable_expression_evaluator::eval_exp2()
{
    char op;
    
    expr* result = eval_exp3();
    
    
    while ((op = *token_) == '+' || op == '-')
    {
        get_token();
        expr* temp = eval_exp3();
        expr* ex = new expr();
        ex->left = result;
        ex->right = temp;
        switch (op) 
        {
        case '-':
            ex->operand = expr::minus_ee;
            result = ex;
            break;
        case '+':
            ex->operand = expr::plus_ee;
            result = ex;
        }
    }

    return  result;
}
// Multiply or divide two factors.
expr* variable_expression_evaluator::eval_exp3()
{
    char op;
    double temp;
    
    expr* result = eval_exp4();
    
    while ((op = *token_) == '*' || op == '/' || op == '%')
    {
        get_token();
        expr* temp = eval_exp4();
        expr* ex = new expr();
        ex->left = result;
        ex->right = temp;
        switch (op) 
        {
        case '*':
            ex->operand = expr::multiply_ee;
            result = ex;
            break;
        case '%':
            ex->operand = expr::modulo_ee;
            result = ex;
            break;
        case '/':
            ex->operand = expr::division_ee;
            result = ex;
            break;
        }
    }

    return result;
}
// Process an exponent.
expr* variable_expression_evaluator::eval_exp4()
{
    
    expr* result = eval_exp5();
    
    while (*token_ == '^')
    {
        get_token();
        expr* temp = eval_exp5();
        expr* ex = new expr();
        ex->left = result;
        ex->right = temp;
        ex->operand = expr::power_ee;
        result = ex;
    }
    return result;
}
// Evaluate a unary + or -.
expr* variable_expression_evaluator::eval_exp5()
{
    char op = 0;
    if ((tok_type_ == UPDATEDELIMITER) && *token_ == '!' || *token_ == '-')
    {
        op = *token_;
        get_token();
    }
    expr* result = eval_exp6();
    expr* ex = new expr();
    ex->left = result;
    
    if (op == '-')
    {
        ex->operand = expr::negation_ee;
        result = ex;
    }
    
    if (op == '!')
    {
        ex->operand = expr::not_ee;
        result = ex;
    }
    
    return result;
}
// Process a function, a parenthesized expression, a value or a variable
expr* variable_expression_evaluator::eval_exp6()
{
    const bool is_func = (tok_type_ == UPDATEFUNCTION);
    char temp_token[80];
    if (is_func)
    {
        strcpy(temp_token, token_);
        get_token();
    }
    if (*token_ == '(') 
    {
        get_token();
        auto result = eval_exp2();
        expr* ex = new expr();
        ex->left = result;
        if (*token_ != ')')
        {
            THROW_LINE("Unbalanced Parentheses");
        }
        if (is_func)
        {
            if (!strcmp(temp_token, "random"))
            {
                ex->operand = expr::random_ee;
                result = ex;
            }
            else if (!strcmp(temp_token, "sqrt"))
            {
                ex->operand = expr::sqrt_ee;
                result = ex;
            }
            else
            {
                THROW_LINE("Unknown Function");
            }
        }
        get_token();
        return result;
    }
    else
        switch (tok_type_)
        {
            case UPDATEVARIABLE:
                {
                    const string var = token_;
                    expr* result = new expr();

                    if (const_local_vars_->count(var))
                    {
                        result->value = const_local_vars_->at(var);
                        result->operand = expr::literal_ee;
                    }
                    else if (const_global_vars_->count(var))
                    {
                        result->value = const_global_vars_->at(var);
                        result->operand = expr::literal_ee;
                    }
                    else if (local_vars_->count(var))
                    {
                        result->variable_id = local_vars_->at(var);
                        result->operand = expr::clock_variable_ee;
                    }
                    else if (global_vars_->count(var))
                    {
                        result->variable_id = global_vars_->at(var);
                        result->operand = expr::clock_variable_ee;
                    }
                    else
                    {
                        string err = "VAR NOT DECLARED. VAR:";
                        err.append(var);
                        err.append(":");
                        THROW_LINE(err)
                    }
                    
                    get_token();
                    return result;
                }
                //result = vars[*token - 'A'];
                
            case UPDATENUMBER:
                {
                    expr* result = new expr();
                    result->value = stod(token_);
                    result->operand = expr::literal_ee;
                    get_token();
                    return result;
                }
            default:
                {
                    string err = "Syntax Error, not known token:";
                    err.append(token_);
                    err.append(":");
                    THROW_LINE(err)
                }
        }
}
// Obtain the next token.
void variable_expression_evaluator::get_token()
{
    tok_type_ = 0;
    char* temp = token_;
    *temp = '\0';
    if (!*exp_ptr_)  // at end of expression
        return;
    while (isspace(*exp_ptr_))  // skip over white space
        ++exp_ptr_; 
    if (strchr("+-*/%^=()!", *exp_ptr_)) 
    {
        tok_type_ = UPDATEDELIMITER;
        *temp++ = *exp_ptr_++;  // advance to next char
    }
    else if (isalpha(*exp_ptr_)) 
    {
        while (!strchr(" +-/*%^=()!\t\r", *exp_ptr_) && (*exp_ptr_))
            *temp++ = *exp_ptr_++;
        while (isspace(*exp_ptr_))  // skip over white space
            ++exp_ptr_;
        tok_type_ = (*exp_ptr_ == '(') ? UPDATEFUNCTION : UPDATEVARIABLE;
    }
    else if (isdigit(*exp_ptr_) || *exp_ptr_ == '.')
    {
        while (!strchr(" +-/*%^=()!\t\r", *exp_ptr_) && (*exp_ptr_))
            *temp++ = *exp_ptr_++;
        tok_type_ = UPDATENUMBER;
    }
    *temp = '\0';
}

expr* variable_expression_evaluator::evaluate_variable_expression(
    const string& input, unordered_map<string, int>* local_vars, unordered_map<string, int>* global_vars,
    unordered_map<string, double>* const_local_vars,unordered_map<string, double>* const_global_vars)
{
    variable_expression_evaluator ob(local_vars, global_vars, const_local_vars, const_global_vars);
    expr* ans = ob.eval_exp(const_cast<char*>(input.substr(0, input.length()).c_str()));
    return ans;
}