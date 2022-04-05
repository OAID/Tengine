#define READ_INPUT_IMAGE(i, base)                                              \
  int in_width_value##i = in_width##i + base;                                  \
  in_width_value##i =                                                          \
      select(in_idx + in_width_value##i, -1,                                   \
             (in_width_value##i < 0 || in_width_value##i >= input_shape.y));   \
  in##i = RI_F(input, SAMPLER, (int2)(in_width_value##i, in_hb_value));

#define CALCULATE_OUTPUT(i)                                                    \
  out##i = mad(in##i.x, weights0, out##i);                                     \
  out##i = mad(in##i.y, weights1, out##i);                                     \
  out##i = mad(in##i.z, weights2, out##i);                                     \
  out##i = mad(in##i.w, weights3, out##i);

#define CALCULATE_OUTPUT_OPT(i)                                                \
  out##i = mad(in_sm##i[local_idx].x, weights0, out##i);                       \
  out##i = mad(in_sm##i[local_idx].y, weights1, out##i);                       \
  out##i = mad(in_sm##i[local_idx].z, weights2, out##i);                       \
  out##i = mad(in_sm##i[local_idx].w, weights3, out##i);

#define GLOBAL_SIZE_2_DIMS                                                     \
  __private const int global_size_dim0, __private const int global_size_dim1,

__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                  \
  if (input1 >= global_size_dim0 || input2 >= global_size_dim1) {              \
    return;                                                                    \
  }

#define GLOBAL_SIZE_3_DIMS                                                     \
  __private const int global_size_dim0, __private const int global_size_dim1,  \
      __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                          \
  if (input1 >= global_size_dim0 || input2 >= global_size_dim1 ||              \
      input3 >= global_size_dim2) {                                            \
    return;                                                                    \
  }

__kernel
#if SET_ATTRIBUTE
    __attribute__((work_group_size_hint(16, 16, 1)))
#endif
    void
    conv_2d(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
            __read_only image2d_t weights,
#ifdef BIAS
            __read_only image2d_t bias,
#endif
            __write_only image2d_t output, __private const int2 input_shape,
            __private const int in_channel_block_length,
            __private const int2 output_shape,
            __private const int2 weights_shape,
            __private const int2 stride_shape,
            __private const int2 padding_shape,
            __private const int2 dilation_shape,
            __private const int out_width_blocks) {

  const int output_channel_width_idx = get_global_id(0);
  const int output_batch_height_idx = get_global_id(1);
  DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

  const int out_channel_block_idx = output_channel_width_idx / out_width_blocks;
  const int out_width_block_idx = output_channel_width_idx % out_width_blocks;

#ifdef BIAS
  FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
#else
  FLOAT4 out0 = (FLOAT4)0;
#endif
  FLOAT4 out1 = out0;
  FLOAT4 out2 = out0;
  FLOAT4 out3 = out0;

  int in_width0 =
      mad24(out_width_block_idx, stride_shape.y << 2, -padding_shape.y);
  int in_width1 = in_width0 + stride_shape.y;
  int in_width2 = in_width0 + stride_shape.y * 2;
  int in_width3 = in_width0 + stride_shape.y * 3;

  const int height_start = mad24((output_batch_height_idx % output_shape.x),
                                 stride_shape.x, -padding_shape.x);
  int in_height_start =
      mad24(select(0, (-height_start + dilation_shape.x - 1) / dilation_shape.x,
                   height_start < 0),
            dilation_shape.x, height_start);
  int in_height_end = min(
      mad24(weights_shape.x, dilation_shape.x, height_start), input_shape.x);

  const int batch_idx =
      mul24((output_batch_height_idx / output_shape.x), input_shape.x);
  const int weights_h_idx =
      mul24(out_channel_block_idx, mul24(weights_shape.y, weights_shape.x)) +
      mul24(select(0, (-height_start + dilation_shape.x - 1) / dilation_shape.x,
                   height_start < 0),
            weights_shape.y);

  FLOAT4 in0, in1, in2, in3;
  FLOAT4 weights0, weights1, weights2, weights3;
  for (int in_channel_block_idx = 0;
       in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) {
    const int in_idx = mul24(in_channel_block_idx, input_shape.y);
    int weights_x_idx = in_channel_block_idx << 2;
    int weights_y_idx = weights_h_idx;
    for (int iy = in_height_start; iy < in_height_end; iy += dilation_shape.x) {
      int in_hb_value = iy + batch_idx;
      for (int w = 0; w < weights_shape.y; w++) {
        int input_width_base = mul24(w, dilation_shape.y);
        READ_INPUT_IMAGE(0, input_width_base);
        READ_INPUT_IMAGE(1, input_width_base);
        READ_INPUT_IMAGE(2, input_width_base);
        READ_INPUT_IMAGE(3, input_width_base);
        weights0 =
            RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx));
        weights1 =
            RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
        weights2 =
            RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
        weights3 =
            RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);
      }
    }
  }

#ifdef RELU
  out0 = fmax(out0, (FLOAT4)0);
  out1 = fmax(out1, (FLOAT4)0);
  out2 = fmax(out2, (FLOAT4)0);
  out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
  out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
  out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
  out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
  out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif


  const int out_x_base = mul24(out_channel_block_idx, output_shape.y);
  int out_x_idx = out_width_block_idx << 2;

  const int remain = output_shape.y - out_x_idx;
  int output_idx = out_x_base + out_x_idx;
  if (remain >= 4) {
    WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
    WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
    WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
    WI_F(output, (int2)(output_idx + 3, output_batch_height_idx), out3);
  } else if (remain == 3) {
    WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
    WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
    WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
  } else if (remain == 2) {
    WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
    WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
  } else if (remain == 1) {
    WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
  }
}
