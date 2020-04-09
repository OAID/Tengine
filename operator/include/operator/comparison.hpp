#ifndef __COMPARISON_HPP__
#define __COMPARISON_HPP__
#include "operator.hpp"
#include "comparison_param.hpp"

namespace TEngine{

class Comparison : public OperatorWithParam<Comparison, ComparisonParam>
{

public:
    Comparison()
    {
        name_ = "Comparison";
    }
    Comparison(const Comparison& src) = default;
    virtual ~Comparison() {};

    void SetSchema(void) override;

    bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout) override;
};

}


#endif