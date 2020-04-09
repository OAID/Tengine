#ifdef ON_A17
#define FUNC_EXTERN extern "C"
#define FUNC_END ;
#else
#define FUNC_EXTERN
#define FUNC_END {}
#endif

FUNC_EXTERN void sgemm_4x12_a17(float* biases, float* input, float* kernel, int kernel_size, float* output, int output_xy,
                              int activation, int layout) FUNC_END

FUNC_EXTERN void sgemm_4x4_a17(float* biases, float* input, float* kernel, int kernel_size, float* output, int output_xy,
                             int activation, int layout) FUNC_END                    
                             