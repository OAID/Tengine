__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                  \
  if (input1 >= global_size_dim0 || input2 >= global_size_dim1) {              \
    return;                                                                    \
  }

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

__kernel void Nearest(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                      __write_only image2d_t output,
                      __private const float height_scale,
                      __private const float width_scale,
                      __private const int input_height,
                      __private const int input_width,
                      __private const int out_height,
                      __private const int out_width) {
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int output_w_idx       = output_cw_idx % out_width;
    const int output_c_block_idx = output_cw_idx / out_width;
    const int output_b_idx       = output_bh_idx / out_height;
    const int output_h_idx       = output_bh_idx % out_height;

    const float scale_height = output_h_idx * height_scale;
    const float scale_width  = output_w_idx * width_scale;
    const int height_lf      = max(0, (int)floor(scale_height));
    const int width_lf       = max(0, (int)floor(scale_width));

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float4 out = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_lf));

    write_imagef(output, (int2)(output_cw_idx, output_bh_idx), out);
}