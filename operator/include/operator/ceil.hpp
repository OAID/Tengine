#ifndef __CEIL_HPP__
#define __CEIL_HPP__

#include "operator.hpp"

namespace TEngine {

class Ceil : public OperatorNoParam<Ceil>
{
public:
    Ceil()
    {
        name_ = "Ceil";
    }
    Ceil(const Ceil& src) = default;
    virtual ~Ceil(){};

    void SetSchema(void) override;

    bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout) override;
};

}   // namespace TEngine

#endif
