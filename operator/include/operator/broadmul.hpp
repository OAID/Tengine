#ifndef __BROADMUL_HPP__
#define __BROADMUL_HPP__

#include "operator.hpp"
#include "bias_param.hpp"

namespace TEngine {

class BroadMul : public OperatorNoParam<BroadMul>
{
public:
    BroadMul()
    {
        name_ = "BroadMul";
    }
    BroadMul(const BroadMul& src) = default;

    void SetSchema(void) override;

    bool InferShape(const std::vector<TEngine::TShape>&, std::vector<TEngine::TShape>&, int layout) override;
};
}    // namespace TEngine

#endif
