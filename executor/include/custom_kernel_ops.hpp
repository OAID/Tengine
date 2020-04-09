#ifndef __CUSTOM_KERNEL_OPS_HPP
#define __CUSTOM_KERNEL_OPS_HPP

#include "custom_kernel.hpp"

namespace TEngine {

class CustomKernelNodeOps : public NodeOps
{
public:
    CustomKernelNodeOps(Node* node, struct custom_kernel_ops* ops)
    {
        k_ops_ = ops;
        k_inputs_ = nullptr;
        k_outputs_ = nullptr;
        input_num_ = 0;
        output_num_ = 0;
    }

    ~CustomKernelNodeOps(void);

    bool OnBind(Node* node) override;
    bool Prerun(Node* node) override;
    bool Reshape(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;

    static NodeOps* NewOps(Node* node, struct custom_kernel_ops* ops);

protected:
    void PrepareOneTensor(Node* node, Tensor* tensor, struct custom_kernel_tensor* t);
    bool PrepareTensors(Node* node);
    void ReleaseTensors(Node* node);

    struct custom_kernel_ops* k_ops_;
    struct custom_kernel_tensor** k_inputs_;
    struct custom_kernel_tensor** k_outputs_;
    int input_num_;
    int output_num_;
};

}    // namespace TEngine

#endif
