#define MIN_VALUE -FLT_MAX

__kernel
void pool_max(const int N,
               __global float* input_data,
               const int input_c,
               const int input_h,
               const int input_w,
               const int output_h,
               const int output_w,
               const int kernel_h,
               const int kernel_w,
               const int stride_h,
               const int stride_w,
               const int pad_h,
               const int pad_w,
               __global float* output_data)
{
    int index = get_global_id(0);

    int output_hw = output_h * output_w;

    int hw_out = index % output_hw;
    int c_out = index / output_hw;
    int h_out = hw_out / output_w;
    int w_out = hw_out % output_w;

    int c_in = c_out;
    int h_in = mad24(h_out, stride_h, - pad_h);
    int w_in = mad24(w_out, stride_w, - pad_w);

    const int h_start = max(h_in, 0);
    const int w_start = max(w_in, 0);
    const int h_end = min(h_in + kernel_h, input_h);
    const int w_end = min(w_in + kernel_w, input_w);

    float maxval = MIN_VALUE;
    input_data = input_data + c_in * input_h * input_w;
    for (int h = h_start; h < h_end; h++)
    {
      for (int w = w_start; w < w_end; w++)
      {
        int maxidx = mad24(h, input_w, w);
        if (input_data[maxidx] > maxval)
        {
          maxval = input_data[maxidx];
        }
      }
    }
    output_data[index] = maxval;
}

__kernel
void pool_avg(const int N,
               __global float* input_data,
               const int input_c,
               const int input_h,
               const int input_w,
               const int output_h,
               const int output_w,
               const int kernel_h,
               const int kernel_w,
               const int stride_h,
               const int stride_w,
               const int pad_h,
               const int pad_w,
               __global float* output_data)
{
    int index = get_global_id(0);

    int output_hw = output_h * output_w;

    int hw_out = index % output_hw;
    int c_out = index / output_hw;
    int h_out = hw_out / output_w;
    int w_out = hw_out % output_w;

    int c_in = c_out;
    int h_in = mad24(h_out, stride_h, - pad_h);
    int w_in = mad24(w_out, stride_w, - pad_w);

    int h_start = max(h_in, 0);
    int w_start = max(w_in, 0);
    int h_end = min(h_in + kernel_h, input_h);
    int w_end = min(w_in + kernel_w, input_w);

    int pool_size = (h_end - h_start) * (w_end - w_start);

    float aveval = 0;
    input_data = input_data + c_in * input_h * input_w;
    for (int h = h_start; h < h_end; h++)
    {
      for (int w = w_start; w < w_end; w++)
      {
        aveval = aveval + input_data[h * input_w + w];
      }
    }
    output_data[index] = aveval / pool_size;
}


__kernel
void global_pool_avg(const int N,
               __global float* input_data,
               const int input_c,
               const int input_h,
               const int input_w,
               const int output_h,
               const int output_w,
               const int kernel_h,
               const int kernel_w,
               const int stride_h,
               const int stride_w,
               const int pad_h,
               const int pad_w,
               __global float* output_data)
{
    int idx = get_global_id(0);

    float aveval = 0;
    input_data = input_data + idx * input_h * input_w;
    for (int h = 0; h < input_h; h++)
    {
      for (int w = 0; w < input_w; w++)
      {
        aveval = aveval + input_data[h * input_w + w];
      }
    }
    output_data[idx] = aveval / (input_h * input_w);
}




