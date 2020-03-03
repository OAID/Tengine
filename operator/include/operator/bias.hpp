#ifndef __BIAS_HPP__
#define __BIAS_HPP__

#include "operator.hpp"
#include "bias_param.hpp"

namespace TEngine {

class Bias : public OperatorWithParam<Bias, BiasParam>
{
public:
    Bias()
    {
        name_ = "Bias";
    }
    Bias(const Bias& src) = default;

    void SetSchema(void) override;

    bool InferShape(const std::vector<TEngine::TShape>&, std::vector<TEngine::TShape>&, int layout) override;
};
}    // namespace TEngine

#endif
