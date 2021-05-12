#ifndef VULKAN_ALLOCATOR_HPP
#define VULKAN_ALLOCATOR_HPP

#include <vulkan/vulkan.h>
#include <pthread.h>
#include <stdlib.h>
#include <list>
#include <vector>
#include <string>
#include "vulkan_platform.hpp"

namespace TEngine {
    
#define MALLOC_ALIGN    16

template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

static inline void* fastMalloc(size_t size)
{
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static inline void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

static inline int TENGINE_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }


class Allocator
{
public:
    virtual ~Allocator();
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};

// class PoolAllocator : public Allocator
// {
// public:
//     PoolAllocator();
//     ~PoolAllocator();

//     // ratio range 0 ~ 1
//     // default cr = 0.75
//     void set_size_compare_ratio(float scr);

//     // release all budgets immediately
//     void clear();

//     virtual void* fastMalloc(size_t size);
//     virtual void fastFree(void* ptr);

// private:
//     Mutex budgets_lock;
//     Mutex payouts_lock;
//     unsigned int size_compare_ratio;// 0~256
//     std::list< std::pair<size_t, void*> > budgets;
//     std::list< std::pair<size_t, void*> > payouts;
// };

// class UnlockedPoolAllocator : public Allocator
// {
// public:
//     UnlockedPoolAllocator();
//     ~UnlockedPoolAllocator();

//     // ratio range 0 ~ 1
//     // default cr = 0.75
//     void set_size_compare_ratio(float scr);

//     // release all budgets immediately
//     void clear();

//     virtual void* fastMalloc(size_t size);
//     virtual void fastFree(void* ptr);

// private:
//     unsigned int size_compare_ratio;// 0~256
//     std::list< std::pair<size_t, void*> > budgets;
//     std::list< std::pair<size_t, void*> > payouts;
// };

class GPUDevice;

class VkBufferMemory
{
public:
    VkBuffer buffer;

    // the base offset assigned by allocator
    size_t offset;
    size_t capacity;

    VkDeviceMemory memory;
    void* mapped_ptr;

    // buffer state, modified by command functions internally
    mutable VkAccessFlags access_flags;
    mutable VkPipelineStageFlags stage_flags;

    // initialize and modified by mat
    int refcount;
};

class VkImageMemory
{
public:
    VkImage image;
    VkImageView imageview;

    // underlying info assigned by allocator
    VkImageType image_type;
    VkImageViewType imageview_type;
    int width;
    int height;
    int depth;
    VkFormat format;

    VkDeviceMemory memory;
    void* mapped_ptr;

    // the base offset assigned by allocator
    size_t bind_offset;
    size_t bind_capacity;

    // image state, modified by command functions internally
    mutable VkAccessFlags access_flags;
    mutable VkImageLayout image_layout;
    mutable VkPipelineStageFlags stage_flags;

    // in-execution state, modified by command functions internally
    mutable int command_refcount;

    // initialize and modified by mat
    int refcount;
};

class VkAllocator
{
public:
    VkAllocator(const GPUDevice* _vkdev);
    virtual ~VkAllocator() { clear(); }
    virtual void clear() {}

    virtual VkBufferMemory* fastMalloc(size_t size) = 0;
    virtual void fastFree(VkBufferMemory* ptr) = 0;
    virtual int flush(VkBufferMemory* ptr);
    virtual int invalidate(VkBufferMemory* ptr);

    virtual VkImageMemory* fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack) = 0;
    virtual void fastFree(VkImageMemory* ptr) = 0;

public:
    const GPUDevice* vkdev;
    uint32_t buffer_memory_type_index;
    uint32_t image_memory_type_index;
    bool mappable;
    bool coherent;

protected:
    VkBuffer create_buffer(size_t size, VkBufferUsageFlags usage);
    VkDeviceMemory allocate_memory(size_t size, uint32_t memory_type_index);
    VkDeviceMemory allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkImage image, VkBuffer buffer);

    VkImage create_image(VkImageType type, int width, int height, int depth, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage);
    VkImageView create_imageview(VkImageViewType type, VkImage image, VkFormat format);
};

class VkBlobAllocator : public VkAllocator
{
public:
    VkBlobAllocator(const GPUDevice* vkdev);
    virtual ~VkBlobAllocator();

public:
    // release all budgets immediately
    virtual void clear();

    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);

    virtual VkImageMemory* fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack);//{ return 0; }
    virtual void fastFree(VkImageMemory* ptr);

protected:
    size_t block_size;
    size_t buffer_offset_alignment;
    size_t bind_memory_offset_alignment;
    std::vector< std::list< std::pair<size_t, size_t> > > buffer_budgets;
    std::vector<VkBufferMemory*> buffer_blocks;
    std::vector< std::list< std::pair<size_t, size_t> > > image_memory_budgets;
    std::vector<VkDeviceMemory> image_memory_blocks;
};

class VkWeightAllocator : public VkAllocator
{
public:
    VkWeightAllocator(const GPUDevice* vkdev);
    virtual ~VkWeightAllocator();

public:
    // release all blocks immediately
    virtual void clear();

public:
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack);//{ return 0; }
    virtual void fastFree(VkImageMemory* ptr);

protected:
    size_t block_size;
    size_t buffer_offset_alignment;
    size_t bind_memory_offset_alignment;
    std::vector<size_t> buffer_block_free_spaces;
    std::vector<VkBufferMemory*> buffer_blocks;
    std::vector<VkBufferMemory*> dedicated_buffer_blocks;
    std::vector<size_t> image_memory_block_free_spaces;
    std::vector<VkDeviceMemory> image_memory_blocks;
    std::vector<VkDeviceMemory> dedicated_image_memory_blocks;
};


class VkStagingAllocator : public VkAllocator
{
public:
    VkStagingAllocator(const GPUDevice* vkdev);
    virtual ~VkStagingAllocator();

public:
    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    virtual void clear();

    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int dims, int w, int h, int c, size_t elemsize, int elempack);//{ return 0; }
    virtual void fastFree(VkImageMemory* ptr);

protected:
    unsigned int size_compare_ratio;// 0~256
    std::list<VkBufferMemory*> buffer_budgets;
};


class VkWeightStagingAllocator : public VkAllocator
{
public:
    VkWeightStagingAllocator(const GPUDevice* vkdev);
    virtual ~VkWeightStagingAllocator();

public:
    virtual VkBufferMemory* fastMalloc(size_t size);
    virtual void fastFree(VkBufferMemory* ptr);
    virtual VkImageMemory* fastMalloc(int /*dims*/, int /*w*/, int /*h*/, int /*c*/, size_t /*elemsize*/, int /*elempack*/) { return 0; }
    virtual void fastFree(VkImageMemory* /*ptr*/) {}

protected:
};

}
#endif
