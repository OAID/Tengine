#ifndef __THRESHOLD_HPP__
#define __THRESHOLD_HPP__

#include "operator.hpp"
#include "threshold_param.hpp"
namespace TEngine{

class Threshold : public OperatorWithParam<Threshold, ThresholdParam>
{
public:
    Threshold()
    {
        name_ = "Threshold";
    }
    Threshold(const Threshold& src) = default;
    virtual ~Threshold() {}

    bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout) override;

    void SetSchema(void) override;
};

}
#endif