#ifndef __LOGICAL_HPP__
#define __LOGICAL_HPP__

#include "operator.hpp"
#include "logical_param.hpp"

namespace TEngine {

class Logical : public OperatorWithParam<Logical, LogicalParam>
{
public:
    Logical()
    {
        name_ = "Logical";
    }
    Logical(const Logical& src) = default;
    virtual ~Logical(){};

    void SetSchema(void) override;

    bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout) override;
};

}    // namespace TEngine

#endif