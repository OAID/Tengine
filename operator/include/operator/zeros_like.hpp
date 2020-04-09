#ifndef __ZEROS_LIKE_HPP__
#define __ZEROS_LIKE_HPP__

#include "operator.hpp"

namespace TEngine {

class ZerosLike : public OperatorNoParam<ZerosLike>
{
public:
    ZerosLike()
    {
        name_ = "ZerosLike";
    }
    ZerosLike(const ZerosLike& src) = default;
    virtual ~ZerosLike(){};

    void SetSchema(void) override;

    bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout) override;
};

}   // namespace TEngine

#endif
