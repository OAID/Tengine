#ifdef ON_A72
#define FUNC_EXTERN extern "C"
#define FUNC_END ;
#else
#define FUNC_EXTERN
#define FUNC_END {}
#endif

FUNC_EXTERN void sgemm_4x16_a72(float* biases, float* input, float* kernel, long kernel_size, float* output,
                               long output_xy, int activatioin, int layout) FUNC_END
FUNC_EXTERN void sgemm_4x4_a72(float* biases, float* input, float* kernel, long kernel_size, float* output,
                              long output_xy, int activatioin, int layout) FUNC_END