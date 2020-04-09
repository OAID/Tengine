#ifndef __INSTANCENORM_HPP__
#define __INSTANCENORM_HPP__

#include "operator.hpp"
#include "instancenorm_param.hpp"

namespace TEngine {

class InstanceNorm : public OperatorWithParam<InstanceNorm, InstanceNormParam>
{
public:
    InstanceNorm(void)
    {
        name_ = "InstanceNorm";
    }
    InstanceNorm(const InstanceNorm&) = default;
    void SetSchema(void) override;
    bool InferShape(const std::vector<TShape>&, std::vector<TShape>&, int layout) override;
};
}
#endif