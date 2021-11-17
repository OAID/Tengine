//#pragma OPENCL EXTENSION cl_amd_printf : enable

__constant sampler_t SAMPLER =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void fc(int global0, __read_only image2d_t input,
                 __read_only image2d_t weight, __read_only image2d_t bias,
                 __write_only image2d_t output,
                 const int input_channel_block4) {
  int x = get_global_id(0);
  // printf("globalid: %d   global0: %d  input_channel_block4:%d \n", x,
  // global0, input_channel_block4);
  int start_x = x << 2;
  if (x < global0) {
    float res0 = 0;
    float res1 = 0;
    float res2 = 0;
    float res3 = 0;
    for (int i = 0; i < input_channel_block4; i++) {
      float4 in0 = read_imagef(input, SAMPLER, (int2)(i, 0));
      float4 weight0 = read_imagef(weight, SAMPLER, (int2)(i, start_x));
      float4 weight1 = read_imagef(weight, SAMPLER, (int2)(i, start_x + 1));
      float4 weight2 = read_imagef(weight, SAMPLER, (int2)(i, start_x + 2));
      float4 weight3 = read_imagef(weight, SAMPLER, (int2)(i, start_x + 3));
      res0 += dot(in0, weight0);
      res1 += dot(in0, weight1);
      res2 += dot(in0, weight2);
      res3 += dot(in0, weight3);
      //   if (x == 0) {
      //     printf("%2.4v4hlf %2.4v4hlf %2.4v4hlf %2.4v4hlf %2.4v4hlf", in0,
      //     res0,
      //            res1, res2, res3);
      //   }
    }
    float4 bias4 = read_imagef(bias, SAMPLER, (int2)(x, 0));
    res0 += bias4.x;
    res1 += bias4.y;
    res2 += bias4.z;
    res3 += bias4.w;

    // if (x == 1) {
    //   printf("%.4f %.4f %.4f %.4f \n", sum, sum1, sum2, sum3);
    // }
    float4 out = (float4)(res0, res1, res2, res3);
    write_imagef(output, (int2)(x, 0), out);
  }
}