#define GLOBAL_SIZE_2_DIMS                                                     \
  __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                  \
  if (input1 >= global_size_dim0 || input2 >= global_size_dim1) {              \
    return;                                                                    \
  }
__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

// convert data from buffer(nchw) to image(b h, ic/4 w ic4)
__kernel void nchw_buffer_to_image(GLOBAL_SIZE_2_DIMS
#ifdef BUFFER_IMAGE_IO_TRANS
                                       __global const float *input_ptr,
#else
                                       __global const FLOAT *input_ptr,
#endif
                                   __private const int height,
                                   __private const int width,
                                   __private const int channels,
                                   __write_only image2d_t output) {
  int image_width_idx = get_global_id(0);
  int image_height_idx = get_global_id(1);

  DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

  const int batch_idx = image_height_idx / height;
  const int height_idx = image_height_idx % height;
  const int width_idx = image_width_idx % width;
  const int channel_4_idx = image_width_idx / width << 2;
  const int buffer_offset =
      ((batch_idx * channels + channel_4_idx) * height + height_idx) * width +
      width_idx;

  const int remain_channel = channels - channel_4_idx;
  const int height_width_size = height * width;
  FLOAT4 output_values = 0;

  if (remain_channel >= 4) {
    int offset = buffer_offset;
    output_values.x = (FLOAT) * (input_ptr + offset);
    offset += height_width_size;
    output_values.y = (FLOAT) * (input_ptr + offset);
    offset += height_width_size;
    output_values.z = (FLOAT) * (input_ptr + offset);
    offset += height_width_size;
    output_values.w = (FLOAT) * (input_ptr + offset);
  } else if (remain_channel == 3) {
    int offset = buffer_offset;
    output_values.x = (FLOAT) * (input_ptr + offset);
    offset += height_width_size;
    output_values.y = (FLOAT) * (input_ptr + offset);
    offset += height_width_size;
    output_values.z = (FLOAT) * (input_ptr + offset);
  } else if (remain_channel == 2) {
    int offset = buffer_offset;
    output_values.x = (FLOAT) * (input_ptr + offset);
    offset += height_width_size;
    output_values.y = (FLOAT) * (input_ptr + offset);
  } else if (remain_channel == 1) {
    int offset = buffer_offset;
    output_values.x = (FLOAT) * (input_ptr + offset);
  }

  WI_F(output, (int2)(image_width_idx, image_height_idx), output_values);
}

// only for debug
// convert data from image(b h, ic/4 w ic4) to buffer(nchw)
__kernel void image_to_nchw_buffer(GLOBAL_SIZE_2_DIMS
#ifdef BUFFER_IMAGE_IO_TRANS
                                       __global float *output, /* nchw */
#else
                                       __global FLOAT *output, /* nchw */
#endif
                                   __private const int height,
                                   __private const int width,
                                   __private const int channels,
                                   __read_only image2d_t input_ptr) {
  int image_width_idx = get_global_id(0);
  int image_height_idx = get_global_id(1);

  DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

  const int batch_idx = image_height_idx / height;
  const int height_idx = image_height_idx % height;
  const int width_idx = image_width_idx % width;
  int channel_4_idx = (image_width_idx / width) * 4;
  int buffer_offset =
      ((batch_idx * channels + channel_4_idx) * height + height_idx) * width +
      width_idx;

#ifdef BUFFER_IMAGE_IO_TRANS
  float4 values = convert_float4(
      RI_F(input_ptr, SAMPLER, (int2)(image_width_idx, image_height_idx)));
#else
  FLOAT4 values =
      RI_F(input_ptr, SAMPLER, (int2)(image_width_idx, image_height_idx));
#endif

  const int height_width_size = height * width;

  const int remain_channel = channels - channel_4_idx;

  if (remain_channel >= 4) {
    int offset = buffer_offset;
    output[offset] = values.x;
    offset += height_width_size;
    output[offset] = values.y;
    offset += height_width_size;
    output[offset] = values.z;
    offset += height_width_size;
    output[offset] = values.w;
  } else if (remain_channel == 3) {
    int offset = buffer_offset;
    output[offset] = values.x;
    offset += height_width_size;
    output[offset] = values.y;
    offset += height_width_size;
    output[offset] = values.z;
  } else if (remain_channel == 2) {
    int offset = buffer_offset;
    output[offset] = values.x;
    offset += height_width_size;
    output[offset] = values.y;
  } else if (remain_channel == 1) {
    int offset = buffer_offset;
    output[offset] = values.x;
  }
}

__kernel void conv2d_filter_buffer_to_image(GLOBAL_SIZE_2_DIMS
                                            #ifdef BUFFER_INP_FP32
                                            __global const float *input_ptr,
                                            #else
                                            __global const FLOAT *input_ptr,
                                            #endif
                                            __private const int output_channel, __private const int2 kernel_shape, __private const int ic_h_w_size,
                                            __private const int height_width_size, __write_only image2d_t output) {
    int image_width_idx  = get_global_id(0); // ic
    int image_height_idx = get_global_id(1); // oc/4 h w

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int input_channel_4_idx  = image_width_idx;
    const int output_channel_4_idx = (image_height_idx / height_width_size) * 4;
    const int height_width_idx     = image_height_idx % height_width_size;
    const int buffer_height_idx    = height_width_idx / kernel_shape.y;
    const int buffer_width_idx     = height_width_idx % kernel_shape.y;

    const int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size +
                              buffer_height_idx * kernel_shape.y + buffer_width_idx;

    FLOAT4 output_values = 0;
    if (output_channel_4_idx < output_channel) {
        const int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.w = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));

        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
        }
    }

    WI_F(output, (int2)(image_width_idx, image_height_idx), output_values);
}

__kernel void dw_filter_buffer_to_image(GLOBAL_SIZE_2_DIMS
                                        #ifdef BUFFER_INP_FP32
                                        __global const float *input_ptr,
                                        #else
                                        __global const FLOAT *input_ptr,
                                        #endif
                                        __private const int4 kernel_shape,
                                        __private const int height_width_size, __write_only image2d_t output) {
    const int image_width_idx  = get_global_id(0);
    const int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    FLOAT4 output_values = 0;
    if (kernel_shape.x == 1) {
        const int input_channel_4_idx = image_height_idx * 4;
        const int buffer_height_idx   = image_width_idx / kernel_shape.w;
        const int buffer_width_idx    = image_width_idx % kernel_shape.w;

        const int buffer_offset =
            mad24(mad24(input_channel_4_idx, kernel_shape.z, buffer_height_idx), kernel_shape.w, buffer_width_idx);

        const int remain_channel = kernel_shape.y - input_channel_4_idx;
        if (input_channel_4_idx < kernel_shape.y) {
            if (remain_channel >= 4) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.w = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 3) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));

            } else if (remain_channel == 2) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 1) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
            }
        }
    }

    WI_F(output, (int2)(image_width_idx, image_height_idx), output_values);
}