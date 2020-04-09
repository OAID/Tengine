#ifndef __CUSTOM_KERNEL_HPP__
#define __CUSTOM_KERNEL_HPP__

#include <string>
#include <vector>

#include "tengine_c_api.h"

#define ATTR_CUSTOM_KERNEL "CUSTOM_KERNEL"
#define ANY_DEVICE_NAME "ANY_DEVICE"

namespace TEngine {

struct NodeOps;
class Node;

struct CustomKernel
{
    std::string dev_name;
    struct custom_kernel_ops* ops;
};

struct CustomKernelList
{
    std::vector<CustomKernel> list;

    struct custom_kernel_ops* GetKernel(const char* dev_name)
    {
        auto ir = FindKernel(dev_name);

        if(ir == list.end())
            return nullptr;

        return ir->ops;
    }

    bool AddKernel(const char* dev_name, struct custom_kernel_ops* ops)
    {
        auto ir = FindKernel(dev_name);

        if(ir != list.end())
            return false;

        CustomKernel k;
        k.dev_name = dev_name;
        k.ops = ops;

        list.emplace_back(k);

        return true;
    }

    bool RemoveKernel(const char* dev_name)
    {
        auto ir = FindKernel(dev_name);

        if(ir == list.end())
            return false;

        list.erase(ir);

        return true;
    }

    typename decltype(list)::iterator FindKernel(const char* dev_name)
    {
        auto start = list.begin();
        auto end = list.end();

        while(start != end)
        {
            if(start->dev_name == dev_name)
                break;

            start++;
        }

        return start;
    }
};

namespace CustomKernelManager {

NodeOps* BindOps(Node* node, const std::string& dev_name);

bool PrepareTensors(Node* node, struct custom_kernel_tensor**& k_inputs, int input_num,
                    struct custom_kernel_tensor**& k_outputs, int output_num);

void ReleaseTensors(Node* node, struct custom_kernel_tensor** k_inputs, int input_num,
                    struct custom_kernel_tensor** k_outputs, int output_num);

}    // namespace CustomKernelManager

}    // namespace TEngine

#endif
