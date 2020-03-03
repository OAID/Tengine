#ifdef ON_A72
#define FUNC_EXTERN extern "C"
#define FUNC_END ;
#else
#define FUNC_EXTERN
#define FUNC_END {}
#endif

FUNC_EXTERN void dw_k3s1p1_a72(float* data, int h, int w, float* kernel, float* output, float* bias, int act) FUNC_END
FUNC_EXTERN void dw_k3s2p1_a72(float* data, int h, int w, float* kernel, float* output, float* bias, int act) FUNC_END
