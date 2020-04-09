#ifdef ON_A17
#define FUNC_EXTERN extern "C"
#define FUNC_END ;
#else
#define FUNC_EXTERN
#define FUNC_END {}
#endif

FUNC_EXTERN void sgemm_4x12_a17(float* biases, float* input, float* kernel, int kernel_size, float* output, 
                               int output_xy, int activation, int layout) FUNC_END

FUNC_EXTERN void sgemm_4x4_a17(float* biases, float* input, float* kernel, int kernel_size, float* output, int output_xy,
                              int activation, int layout) FUNC_END

FUNC_EXTERN void direct_k3s1p1_4x4_a17(float* biases, float* input, float* kernel, float* output, int input_chan,
                                      int input_w, int input_h, int activation) FUNC_END

FUNC_EXTERN void direct_k3s1p1_4x12_a17(float* biases, float* input, float* kernel, float* output, int input_chan,
                                       int input_w, int input_h, int activation) FUNC_END

FUNC_EXTERN void direct_k1s1p0_4x12_a17(float* biases, float* input, float* kernel, float* output, int output_xy,
                                       int c_in, int activation) FUNC_END

FUNC_EXTERN void direct_k1s1p0_4x4_a17(float* biases, float* input, float* kernel, float* output, int output_xy,
                                      int c_in, int activation) FUNC_END
