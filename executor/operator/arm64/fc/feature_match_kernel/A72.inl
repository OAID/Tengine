#ifdef ON_A72
#define FUNC_EXTERN extern "C"
#define FUNC_END ;
#else
#define FUNC_EXTERN
#define FUNC_END {}
#endif

FUNC_EXTERN void sgemv_1x8_a72(float* biases, float* input, float* kernel, long kernel_size, float* output) FUNC_END

FUNC_EXTERN void sgemv_1x2_a72(float* biases, float* input, float* kernel, long kernel_size, float* output) FUNC_END