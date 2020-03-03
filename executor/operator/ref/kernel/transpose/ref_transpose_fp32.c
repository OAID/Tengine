void transpose2d(float* input, float* output, const ref_transpose_param* param){
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]]; 
    int in_dim1 = param->in_dims[1];
    int inStride[2];
    inStride[0] = in_dim1;
    inStride[1] = 1;
    
    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];

    for(int n = 0; n < out_dim0; n++){ // 1
        for(int h = 0; h < out_dim1; h++){ // 1
            output[n*out_dim1 + h] =
                input[n*stride0 + h*stride1];
        }
    }
    return ;
}
void transpose3d(float* input, float* output, const ref_transpose_param* param){
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]];
    int out_dim2 = param->in_dims[param->permute[2]];


    int in_dim1 = param->in_dims[1];
    int in_dim2 = param->in_dims[2];
    

    int inStride[3];
    inStride[0] = in_dim1*in_dim2;
    inStride[1] = in_dim2;
    inStride[2] = 1;
    
    int outStride0 = out_dim1*out_dim2;
    int outStride1 = out_dim2;
    

    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];
    int stride2 = inStride[param->permute[2]];

    for(int n = 0; n < out_dim0; n++){ // 1
        for(int h = 0; h < out_dim1; h++){ // 1
            for(int w = 0; w < out_dim2; w++){ // 2
                    output[n*outStride0 + h*outStride1 + w] =
                        input[n*stride0 + h*stride1 + w*stride2 ];
            }
        }
    }
    return ;
}
void transpose4d(float* input, float* output, const ref_transpose_param* param){
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]];
    int out_dim2 = param->in_dims[param->permute[2]];
    int out_dim3 = param->in_dims[param->permute[3]];

    int in_dim1 = param->in_dims[1];
    int in_dim2 = param->in_dims[2];
    int in_dim3 = param->in_dims[3];

    int inStride[4];
    inStride[0] = in_dim1*in_dim2*in_dim3;
    inStride[1] = in_dim2*in_dim3;
    inStride[2] = in_dim3;
    inStride[3] = 1;

    int outStride0 = out_dim1*out_dim2*out_dim3;
    int outStride1 = out_dim2*out_dim3;
    int outStride2 = out_dim3;    

    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];
    int stride2 = inStride[param->permute[2]];
    int stride3 = inStride[param->permute[3]];

    for(int n = 0; n < out_dim0; n++){ // 1
        for(int h = 0; h < out_dim1; h++){ // 1
            for(int w = 0; w < out_dim2; w++){ // 2
                for(int c = 0; c < out_dim3; c++){ // 2
                    output[n*outStride0 + h*outStride1 + w*outStride2 + c] =
                        input[n*stride0 + h*stride1 + w*stride2 + c*stride3];
                }
            }
        }
    }
    return ;
}
void transpose5d(float* input, float* output, const ref_transpose_param* param){
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]];
    int out_dim2 = param->in_dims[param->permute[2]];
    int out_dim3 = param->in_dims[param->permute[3]];
    int out_dim4 = param->in_dims[param->permute[4]];

    int in_dim1 = param->in_dims[1];
    int in_dim2 = param->in_dims[2];
    int in_dim3 = param->in_dims[3];
    int in_dim4 = param->in_dims[4];

    int inStride[5];
    inStride[0] = in_dim1*in_dim2*in_dim3*in_dim4;
    inStride[1] = in_dim2*in_dim3*in_dim4;
    inStride[2] = in_dim3*in_dim4;
    inStride[3] = in_dim4;
    inStride[4] = 1;

    int outStride0 = out_dim1*out_dim2*out_dim3*out_dim4;
    int outStride1 = out_dim2*out_dim3*out_dim4;
    int outStride2 = out_dim3*out_dim4;
    int outStride3 = out_dim4;    

    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];
    int stride2 = inStride[param->permute[2]];
    int stride3 = inStride[param->permute[3]];
    int stride4 = inStride[param->permute[4]];

    for(int n = 0; n < out_dim0; n++){ // 1
        for(int h = 0; h < out_dim1; h++){ // 1
            for(int w = 0; w < out_dim2; w++){ // 2
                for(int c = 0; c < out_dim3; c++){ // 2
                    for(int x = 0; x < out_dim4; x++){
                        output[n*outStride0 + h*outStride1 + w*outStride2 + c*outStride3 + x] =
                            input[n*stride0 + h*stride1 + w*stride2 + c*stride3 + x*stride4];
                    }
                }
            }
        }
    }
    return ;
}
void transpose6d(float* input, float* output, const ref_transpose_param* param){
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]];
    int out_dim2 = param->in_dims[param->permute[2]];
    int out_dim3 = param->in_dims[param->permute[3]];
    int out_dim4 = param->in_dims[param->permute[4]];
    int out_dim5 = param->in_dims[param->permute[5]];


    int in_dim1 = param->in_dims[1];
    int in_dim2 = param->in_dims[2];
    int in_dim3 = param->in_dims[3];
    int in_dim4 = param->in_dims[4];
    int in_dim5 = param->in_dims[5];


    int inStride[6];
    inStride[0] = in_dim1*in_dim2*in_dim3*in_dim4*in_dim5;
    inStride[1] = in_dim2*in_dim3*in_dim4*in_dim5;
    inStride[2] = in_dim3*in_dim4*in_dim5;
    inStride[3] = in_dim4*in_dim5;
    inStride[4] = in_dim5;
    inStride[5] = 1;

    int outStride0 = out_dim1*out_dim2*out_dim3*out_dim4*out_dim5;
    int outStride1 = out_dim2*out_dim3*out_dim4*out_dim5;
    int outStride2 = out_dim3*out_dim4*out_dim5;
    int outStride3 = out_dim4*out_dim5;
    int outStride4 = out_dim5;    

    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];
    int stride2 = inStride[param->permute[2]];
    int stride3 = inStride[param->permute[3]];
    int stride4 = inStride[param->permute[4]];
    int stride5 = inStride[param->permute[5]];

    for(int n = 0; n < out_dim0; n++){ // 1
        for(int h = 0; h < out_dim1; h++){ // 1
            for(int w = 0; w < out_dim2; w++){ // 2
                for(int c = 0; c < out_dim3; c++){ // 2
                    for(int x = 0; x < out_dim4; x++){
                        for(int y = 0; y < out_dim5; y++){
                            output[n*outStride0 + h*outStride1 + w*outStride2 + c*outStride3 + x*outStride4 + y] =
                                input[n*stride0 + h*stride1 + w*stride2 + c*stride3 + x*stride4 + y*stride5];
                        }
                    }
                }
            }
        }
    }
    return ;
}
static int ref_transpose_fp32(float* input, float* output, const ref_transpose_param* param)
{
    switch(param->dims){
        case 2:
            transpose2d(input, output, param);
            break;
        case 3:
            transpose3d(input, output, param);
            break;
        case 4:
            transpose4d(input, output, param);
            break;
        case 5:
            transpose5d(input, output, param);
            break;
        case 6:
            transpose6d(input, output, param);
            break;
        default:
            break;
    }
    return 0;
}
