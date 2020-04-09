#include "executor/operator/ref/kernel/elu/ref_elu_kernel.h"

#include <cfloat>
#include <cmath>
#include <cstdio>


int main(int argc, char *argv[])
{
    elu_param elu_param;
    elu_param.scale = 1.0f;
    elu_param.zero_point = 0;

    float input[4] = {0.0f, 1.0f, -1.0f, -5.0f};
    float output[4]{0.0f};
    float value[4] = {0.0f, 1.0f, -0.63212055883f, -0.99326205300f};

    float epsilon = FLT_EPSILON * 10;

    ref_elu_fp32(input, output, 4, &elu_param);

    for (size_t i = 0; i < 4; i++)
    {
        if (std::fabs(value[i] - output[i]) <= epsilon)
        {
            //printf("%0.8f <= %0.8f.\n", std::fabs(value[i] - output[i]), epsilon);
            continue;
        }

        printf("Test ELU error: ref=%0.6f vs. out=%0.6f does not less than epsilon=%0.6f\n", value[i], output[i], epsilon);
        printf("%0.8f > %0.8f.\n", std::fabs(value[i] - output[i]), epsilon);

        return -1;
    }

    return 0;
}
