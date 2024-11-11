#ifndef MEMORY_ALLOCATOR_CU
#define MEMORY_ALLOCATOR_CU

#include "../../include/common_macros.h"
#include <list>
class memory_allocator
{
private:
    std::list<void*> cuda_free_list_;
    std::list<void*> free_list_;
    size_t cuda_allocation_size_;
    size_t host_allocation_size_;

public:
    const bool use_cuda;

    explicit memory_allocator(const bool use_cuda);

    template<typename T>
    cudaError allocate(T** p, const size_t size);

    template<typename T>
    cudaError allocate_cuda(T** p, const size_t size);

    template<typename T>
    cudaError allocate_host(T** p, const size_t size);

    template<typename T>
    cudaError allocate_and_copy(T** dest, const T* source, const unsigned amount);
    template <class T>
    cudaError memery_set(T* p, int memory_value, int size);

    void add_to_host_freelist(void* p, size_t size = 0);

    size_t get_cuda_memory_usage() const
    {
        return this->cuda_allocation_size_;
    }

    void free_allocations();
};

template <typename T>
cudaError memory_allocator::allocate(T** p, const size_t size)
{
    if(size == 0)
    {
        *p = nullptr;
        return cudaSuccess;
    }

    if(this->use_cuda)
        return allocate_cuda(p, size);
    return allocate_host(p, size);
}

template <typename T>
cudaError memory_allocator::allocate_cuda(T** p, const size_t size)
{
    const cudaError e = cudaMalloc(p, size);
    this->cuda_free_list_.push_back(*p);
    this->cuda_allocation_size_ += size;
    return e;

}

template <typename T>
cudaError memory_allocator::allocate_host(T** p, const size_t size)
{
    *p = static_cast<T*>(malloc(size));
    this->free_list_.push_back(*p);
    this->host_allocation_size_ += size;
    return cudaSuccess;

}

template <typename T>
cudaError memory_allocator::allocate_and_copy(T** dest, const T* source, const unsigned amount)
{
    const size_t size = sizeof(T)*amount;
    const cudaMemcpyKind kind = use_cuda ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    if(amount == 0)
    {
        *dest = nullptr;
        return cudaSuccess;
    }

    cudaError e = this->allocate(dest, size);
    if(e != cudaSuccess) return e;

    e = cudaMemcpy(*dest, source, size, kind);
    return e;
}

template <typename T>
cudaError memory_allocator::memery_set(T* p, int memory_value, int size)
{
    const size_t mem_size = sizeof(T)*size;
    if(use_cuda)
    {
        return cudaMemset(p, memory_value, mem_size);
    }
    else
    {
        memset(p, memory_value, mem_size);
        return cudaSuccess;
    }
}

#endif