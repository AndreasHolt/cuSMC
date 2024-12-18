//
// Created by andwh on 10/12/2024.
//

#include <cfloat>
#include "state/shared_model_state.cuh"
#include "state/shared_run_state.cuh"
#include "../automata_parser/uppaal_xml_parser.h"

__device__ double evaluate_expression_node_coalesced(const expr* expr, SharedBlockMemory* shared, double* value_stack, int stack_top)
{
    double v1, v2;
    switch (expr->operand) {
    case expr::literal_ee:
        return fetch_expr_value(expr);
    case expr::clock_variable_ee:
        return shared->variables[expr->variable_id].value;
    case expr::random_ee:
        printf("Error: Unsupported Random operater.\n");
        return 1.0;
    case expr::plus_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return v1 + v2;
    case expr::minus_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return v1 - v2;
    case expr::multiply_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return v1 * v2;
    case expr::division_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return v1 / v2;
    case expr::power_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return pow(v1, v2);
    case expr::negation_ee:
        v1 = value_stack[--stack_top];
        return -v1;
    case expr::sqrt_ee:
        v1 = value_stack[--stack_top];
        return sqrt(v1);
    case expr::modulo_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return static_cast<double>(static_cast<int>(v1) % static_cast<int>(v2 + DBL_EPSILON));
    case expr::and_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return static_cast<double>(abs(v1) > DBL_EPSILON && abs(v2) > DBL_EPSILON);
    case expr::or_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return static_cast<double>(abs(v1) > DBL_EPSILON || abs(v2) > DBL_EPSILON);
    case expr::less_equal_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return v1 <= v2;
    case expr::greater_equal_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return v1 >= v2;
    case expr::less_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return v1 < v2;
    case expr::greater_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return v1 > v2;
    case expr::equal_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return abs(v1 - v2) <= DBL_EPSILON;
    case expr::not_equal_ee:
        v2 = value_stack[--stack_top];
        v1 = value_stack[--stack_top];
        return abs(v1 - v2) > DBL_EPSILON;
    case expr::not_ee:
        v1 = value_stack[--stack_top];
        return (abs(v1) < DBL_EPSILON);
    case expr::conditional_ee:
        v1 = value_stack[--stack_top];
        stack_top--;
        return v1;
    case expr::compiled_ee:
    case expr::pn_compiled_ee:
    case expr::pn_skips_ee: return 0.0;
    }
    return 0.0;
}

__device__ double evaluate_pn_expr_coalesced(const expr* pn, SharedBlockMemory* shared)
{
    int stack_top = 0;
    double value_stack[10];

    for (int i = 1; i < pn->length; ++i)
    {
        const expr* current = &pn[i];
        if(current->operand == expr::pn_skips_ee)
        {
            if(abs(value_stack[--stack_top]) > DBL_EPSILON)
            {
                i += current->length;
            }
            continue;
        }

        const double value = evaluate_expression_node_coalesced(current, shared, value_stack, stack_top);
        value_stack[stack_top++] = value;
    }

    return value_stack[--stack_top];
}

__device__ double evaluate_expression(const expr *e, SharedBlockMemory *shared) {
    if (e == nullptr) {
        printf("Warning: Null expression in evaluate_expression\n");
        return 0.0;
    }

    // Get operand using our helper
    int op = fetch_expr_operand(e);

    if (op == expr::pn_compiled_ee) {
        return evaluate_pn_expr_coalesced(e, shared);
    }

    // Handle non-PN expressions
    if constexpr (VERBOSE) {
        printf("DEBUG: Evaluating non-pn expression with operator %d\n", op);
    }

    switch (op) {
        case expr::literal_ee:
            return fetch_expr_value(e);

        case expr::clock_variable_ee: {
            int var_id = e->variable_id; // Union access, single read

            if (var_id < MAX_VARIABLES) {
                return shared->variables[var_id].value;
            }
            printf("Warning: Invalid variable ID %d in expression\n", var_id);
            return 0.0;
        }

        case expr::plus_ee: {
            const expr *left = fetch_expr(e->left);
            const expr *right = fetch_expr(e->right);
            if (left && right) {
                double left_val = evaluate_expression(left, shared);
                double right_val = evaluate_expression(right, shared);
                printf("Left and right values: %f, %f\n", left_val, right_val);
                if constexpr (EXPR_VERBOSE) {
                    printf("DEBUG: Plus operation: %f + %f = %f\n",
                           left_val, right_val, left_val + right_val);
                }
                return left_val + right_val;
            }
            break;
        }

        case expr::minus_ee: {
            const expr *left = fetch_expr(e->left);
            const expr *right = fetch_expr(e->right);
            if (left && right) {
                double left_val = evaluate_expression(left, shared);
                double right_val = evaluate_expression(right, shared);
                return left_val - right_val;
            }
            break;
        }

        case expr::multiply_ee: {
            const expr *left = fetch_expr(e->left);
            const expr *right = fetch_expr(e->right);
            if (left && right) {
                double left_val = evaluate_expression(left, shared);
                double right_val = evaluate_expression(right, shared);
                return left_val * right_val;
            }
            break;
        }

        case expr::division_ee: {
            const expr *left = fetch_expr(e->left);
            const expr *right = fetch_expr(e->right);
            if (left && right) {
                double left_val = evaluate_expression(left, shared);
                double right_val = evaluate_expression(right, shared);

                printf("Left and right values: %f, %f\n", left_val, right_val);

                if (right_val == 0.0) {
                    printf("Warning: Division by zero\n");
                    return 0.0;
                }
                double result = left_val / right_val;

                if (result == 0.0) {
                    // TODO: Check that this is used for a rate
                    printf("Warning: Expression evaluates to zero, cannot be used as exponential rate\n");
                    return DBL_MIN; // Minimum valid rate
                }

                return left_val / right_val;
            }
            break;
        }

        case expr::pn_skips_ee:
            if (e->left) {
                return evaluate_expression(fetch_expr(e->left), shared);
            }
            break;

        default:
            printf("Warning: Unhandled operator %d in expression\n", op);
            break;
    }

    return 0.0;
}