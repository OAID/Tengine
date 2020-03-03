#ifndef __SQUARED_DIFFERENCE_HPP__
#define __SQUARED_DIFFERENCE_HPP__

#include "operator.hpp"

namespace TEngine {

class SquaredDifference : public OperatorNoParam<SquaredDifference>
{
public:
    SquaredDifference()
    {
        name_ = "SquaredDifference";
    }
    SquaredDifference(const SquaredDifference& src) = default;
    virtual ~SquaredDifference(){};

    void SetSchema(void) override;

    bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout) override;
};

}   // namespace TEngine

#endif