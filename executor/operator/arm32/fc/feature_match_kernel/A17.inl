#ifdef ON_A17
#define FUNC_EXTERN extern "C"
#define FUNC_END ;
#else
#define FUNC_EXTERN
#define FUNC_END {}
#endif

FUNC_EXTERN void sgemv_1x8_a17(float* biases, float* input, float* kernel, int kernel_size, float* output) FUNC_END

FUNC_EXTERN void sgemv_1x2_a17(float* biases, float* input, float* kernel, int kernel_size, float* output) FUNC_END