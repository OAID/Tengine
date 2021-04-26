
__kernel
void depthwise_conv(const int N,
               __global float* input_data,
               __global float* weight_data,
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

    const int h_in_start = max(h_in, 0);
    const int w_in_start = max(w_in, 0);
    const int h_in_end = min(h_in + kernel_h, input_h);
    const int w_in_end = min(w_in + kernel_w, input_w);

    int h_weight_start = h_in_start - h_in;
    int w_weight_start = w_in_start - w_in;

    float sumval = 0;
    input_data = input_data + c_in * input_h * input_w;
    weight_data = weight_data + c_in * kernel_h * kernel_w;

    int h0 = h_weight_start;
    for (int h = h_in_start; h < h_in_end; h++)
    {
      int w0 = w_weight_start;
      for (int w = w_in_start; w < w_in_end; w++)
      {
        int input_idx = mad24(h, input_w, w);
        int weight_idx = mad24(h0, kernel_w, w0);
        sumval = mad(input_data[input_idx], weight_data[weight_idx], sumval);
        w0 ++;
      }
      h0 ++;
    }
    output_data[index] = sumval;
}