#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel
void im2col(__global const float* data_im,
            const int col_chw,
            const int height, const int width,
            const int kernel_c, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int height_col, const int width_col,
            __global float* col_data) {
  int index = get_global_id(0); // [0, col_chw)

  int kernel_hw = kernel_h * kernel_w;
  int kernel_chw = kernel_c * kernel_hw;

  if(index < col_chw) {
    int h_out = index / kernel_chw;
    int w_out = index % kernel_chw;

    int h_step = h_out / width_col;
    int w_step = h_out % width_col;

    int hw_kernel = w_out % kernel_hw;
    int h_kernel = hw_kernel / kernel_w;
    int w_kernel = hw_kernel % kernel_w;

    int c_in = w_out / kernel_hw;
    int h_in = mad24(h_step, stride_h, - pad_h + h_kernel);
    int w_in = mad24(w_step, stride_w, - pad_w + w_kernel);

    col_data[index] = (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width)
        ? data_im[c_in * height * width + h_in * width + w_in]
        : 0;
  }
}

