#pragma once

#include <unordered_set>

#include "../engine/Domain.h"

class visitor
{
protected:
    std::unordered_set<const void*> visit_set_;
    bool has_visited(const void* p);

    static void accept(const network* a, visitor* v);
    static void accept(const node* n, visitor* v);
    static void accept(const edge* e, visitor* v);
    static void accept(const constraint* c, visitor* v);
    static void accept(const clock_var* c, visitor* v);
    static void accept(const update* u, visitor* v);
    static void accept(const expr* ex, visitor* v);
    
public:
    virtual ~visitor() = default;
    virtual void visit(network* a) = 0;
    virtual void visit(node* n) = 0;
    virtual void visit(edge* e) = 0;
    virtual void visit(constraint* c) = 0;
    virtual void visit(clock_var* cv) = 0 ;
    virtual void visit(update* u) = 0;
    virtual void visit(expr* u) = 0;

    virtual void clear();
};
