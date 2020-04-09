#ifndef __HARDSIGMOID_HPP__
#define __HARDSIGMOID_HPP__

#include "operator.hpp"
#include "hardsigmoid_param.hpp"

namespace TEngine{

class Hardsigmoid : public OperatorWithParam<Hardsigmoid, HardsigmoidParam>
{
public:
    Hardsigmoid(void)
    {
        name_ = "Hardsigmoid";
    }

    Hardsigmoid(const Hardsigmoid&) = default;
    void SetSchema(void) override;
    bool InferShape(const std::vector<TShape>&, std::vector<TShape>&, int layout) override;
 
};

}
#endif