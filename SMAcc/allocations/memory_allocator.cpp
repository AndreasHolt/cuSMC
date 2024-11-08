#include "memory_allocator.h"

memory_allocator::memory_allocator(const bool use_cuda): use_cuda(use_cuda)
{
    this->free_list_ = std::list<void*>();
    this->cuda_free_list_ = std::list<void*>();
    this->cuda_allocation_size_ = 0;
    this->host_allocation_size_ = 0;
}

void memory_allocator::add_to_host_freelist(void* p, size_t size)
{
    this->free_list_.push_back(p);
    this->host_allocation_size_ += size;
}

void memory_allocator::free_allocations()
{
    this->cuda_allocation_size_ = 0;
    this->host_allocation_size_ = 0;
    for (const auto p : this->free_list_)
    {
        free(p);
    }
    for (const auto p : this->cuda_free_list_)
    {
        cudaFree(p);
    }
}