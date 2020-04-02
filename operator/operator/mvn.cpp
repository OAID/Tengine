#include "operator/mvn.hpp"

namespace TEngine{

void MVN::SetSchema(void)
{
    Input({"input:float32"})
    .Output({"output:float32"})
    .SetAttr("normalize_variance",0)
    .SetAttr("across", 0)
    .SetAttr("eps", 0.0001f);
}

}