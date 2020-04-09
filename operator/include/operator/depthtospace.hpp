#ifndef __DEPTHTOSPACE_HPP__
#define __DEPTHTOSPACE_HPP__

#include "operator.hpp"
#include "operator/depthtospace_param.hpp"

namespace TEngine{

class DepthToSpace : public OperatorWithParam<DepthToSpace, DepthToSpaceParam>
{
public:
    DepthToSpace()
    {
        name_ = "DepthToSpace";
    }

    DepthToSpace(const DepthToSpace& src) = default;

    virtual ~DepthToSpace(){}

    void SetSchema(void) override;

    bool InferShape(const std::vector<TEngine::TShape>&, std::vector<TEngine::TShape>&, int layout) override;

};

}
#endif