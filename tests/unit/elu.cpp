#include "executor/operator/common/elu.cpp"

#include <cfloat>
#include <cmath>


int main(int argc, char *argv[]) {
    TEngine::EluOps* ops;

    ops = new TEngine::EluImpl::EluOps;

    float input[4] = {0.0f, 1.0f, -1.0f, -5.0f};
    float output[4]{0.0f};
    float value[4] = {0.0f, 1.0f, -0.63212055883f, -0.99326205300f};

    float epsilon = FLT_EPSILON * 10;

    ops->kernel_run((float*)(&input), (float*)(&output), 4);

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
