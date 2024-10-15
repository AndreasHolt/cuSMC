#include "macro.h"

template<typename T>
class my_stack
{
private:
    T* store_;
    int size_;
    int count_;
public:
    CPU GPU explicit my_stack(T* store, int size)
    {
        this->store_ = store;
        this->size_ = size;
        this->count_ = 0;
    }

    CPU GPU void push(T& t)
    {
        store_[count_++] = t;
    }
    CPU GPU void push_val(T t)
    {
        store_[count_++] = t;
    }

    CPU GPU T pop()
    {
        // if(this->count_ <= 0) printf("stack is empty, cannot pop! PANIC!");
        return this->store_[--this->count_];
    }

    CPU GPU T peak()
    {
        // if(this->count_ <= 0) printf("stack is empty, cannot peak! PANIC!");
        return this->store_[this->count_ - 1];
    }

    CPU GPU int count() const
    {
        return this->count_;
    }

    CPU GPU void clear()
    {
        this->count_ = 0;
    }
};
