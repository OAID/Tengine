#ifndef __SPACETODEPTH_HPP__
#define __SPACETODEPTH_HPP__

#include "operator.hpp"
#include "operator/spacetodepth_param.hpp"

namespace TEngine{

class SpaceToDepth : public OperatorWithParam<SpaceToDepth, SpaceToDepthParam>
{
public:
    SpaceToDepth()
    {
        name_ = "SpaceToDepth";
    }

    SpaceToDepth(const SpaceToDepth& src) = default;

    virtual ~SpaceToDepth(){}

    void SetSchema(void) override;

    bool InferShape(const std::vector<TEngine::TShape>&, std::vector<TEngine::TShape>&, int layout) override;

};

}
#endif