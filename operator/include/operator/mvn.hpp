#ifndef __MVN_HPP__
#define __MVN_HPP__

#include "operator.hpp"
#include "operator/mvn_param.hpp"

namespace TEngine{
class MVN : public OperatorWithParam<MVN, MVNParam>
{
public:
    MVN()
    {
        name_ = "MVN";
    }
    MVN(const MVN& src) = default;
    virtual ~MVN(){};

    void SetSchema(void) override;
};

}
#endif