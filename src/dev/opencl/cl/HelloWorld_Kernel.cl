__kernel void helloworld(__global float* in, __global float* out)
 {
     int num = get_global_id(0);
 //   out[num] = in[num] + 1;
     out[num] = in[num] / 2.4 *(in[num]/6) ;
 }