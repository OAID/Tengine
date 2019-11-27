void sum_4d_ax0_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param);
void sum_4d_ax1_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param);
void sum_4d_ax2_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param);
void sum_4d_ax3_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param);
void sum_3d_ax0_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void sum_3d_ax1_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void sum_3d_ax2_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void sum_2d_ax0_uint8(int dim1, int dim2, float* tmp, float* tmp_0);
void sum_2d_ax1_uint8(int dim1, int dim2, float* tmp, float* tmp_1);

void mean_4d_ax0_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param);
void mean_4d_ax1_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param);
void mean_4d_ax2_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param);
void mean_4d_ax3_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param);
void mean_3d_ax0_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_01);
void mean_3d_ax1_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_02);
void mean_3d_ax2_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_03);
void mean_2d_ax0_uint8(int dim1, int dim2, float* tmp, float* tmp_0);
void mean_2d_ax1_uint8(int dim1, int dim2, float* tmp, float* tmp_1);

static int ref_reduce_uint8(uint8_t* data, uint8_t* out_data, int dim0, int dim1, int dim2, int dim3, int out_size,
                            reduce_param* param)
{
    int offset = 0;
    float* tmp = ( float* )malloc(sizeof(float) * out_size);
    memset(tmp, 0, sizeof(float) * out_size);
    int param_dim0 = param->param_dim[0];
    int param_dim1 = param->param_dim[1];
    int param_dim2 = param->param_dim[2];
    int param_dim3 = param->param_dim[3];
    // reduce sum
    if(param->type == 0)
    {
        if((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
           (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            for(int n = 0; n < dim0; n++)
            {
                for(int h = 0; h < dim1; h++)
                {
                    for(int w = 0; w < dim2; w++)
                    {
                        for(int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
                            tmp[0] += real_input0;
                        }
                    }
                }
            }
        }
        else if(param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sum_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp, param);
        }
        else if(param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sum_4d_ax1_uint8(dim0, dim1, dim2, dim3, data, tmp, param);
        }
        else if(param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sum_4d_ax2_uint8(dim0, dim1, dim2, dim3, data, tmp, param);
        }
        else if(param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            sum_4d_ax3_uint8(dim0, dim1, dim2, dim3, data, tmp, param);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            sum_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_01, param);
            sum_3d_ax0_uint8(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            sum_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_02, param);
            sum_3d_ax1_uint8(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            sum_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_03, param);
            sum_3d_ax2_uint8(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            sum_4d_ax1_uint8(dim0, dim1, dim2, dim3, data, tmp_12, param);
            sum_3d_ax1_uint8(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            sum_4d_ax1_uint8(dim0, dim1, dim2, dim3, data, tmp_13, param);
            sum_3d_ax2_uint8(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            sum_4d_ax2_uint8(dim0, dim1, dim2, dim3, data, tmp_23, param);
            sum_3d_ax2_uint8(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if(param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                     (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                     (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                     (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                     (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                     (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            sum_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_01, param);
            sum_3d_ax0_uint8(dim1, dim2, dim3, tmp_0, tmp_01);
            sum_2d_ax0_uint8(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if(param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                     (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                     (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                     (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                     (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                     (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            sum_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_01, param);
            sum_3d_ax0_uint8(dim1, dim2, dim3, tmp_1, tmp_01);
            sum_2d_ax1_uint8(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if(param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                     (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                     (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                     (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                     (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                     (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            sum_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_02, param);
            sum_3d_ax1_uint8(dim1, dim2, dim3, tmp_1, tmp_02);
            sum_2d_ax1_uint8(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if(param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                     (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                     (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                     (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                     (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                     (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            sum_4d_ax1_uint8(dim0, dim1, dim2, dim3, data, tmp_12, param);
            sum_3d_ax1_uint8(dim0, dim2, dim3, tmp_1, tmp_12);
            sum_2d_ax1_uint8(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // reduce mean
    else if(param->type == 1)
    {
        if((param_dim0 == -2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2) ||
           (param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2 && param_dim3 == 3))
        {
            float s_tmp = 0.f;
            for(int n = 0; n < dim0; n++)
            {
                for(int h = 0; h < dim1; h++)
                {
                    for(int w = 0; w < dim2; w++)
                    {
                        for(int c = 0; c < dim3; c++)
                        {
                            // nhwc
                            offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                            float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
                            s_tmp += real_input0;
                        }
                    }
                }
            }
            tmp[0] = s_tmp / (dim0 * dim1 * dim2 * dim3);
        }
        else if(param_dim0 == 0 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            mean_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp, param);
        }
        else if(param_dim0 == 1 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            mean_4d_ax1_uint8(dim0, dim1, dim2, dim3, data, tmp, param);
        }
        else if(param_dim0 == 2 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            mean_4d_ax2_uint8(dim0, dim1, dim2, dim3, data, tmp, param);
        }
        else if(param_dim0 == 3 && param_dim1 == -2 && param_dim2 == -2 && param_dim3 == -2)
        {
            mean_4d_ax3_uint8(dim0, dim1, dim2, dim3, data, tmp, param);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 1 && param_dim1 == 0) || (param_dim0 == 0 && param_dim1 == 1)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);
            mean_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_01, param);
            mean_3d_ax0_uint8(dim1, dim2, dim3, tmp, tmp_01);

            free(tmp_01);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 0 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);
            mean_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_02, param);
            mean_3d_ax1_uint8(dim1, dim2, dim3, tmp, tmp_02);

            free(tmp_02);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 0 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 0)))
        {
            // reduce on axis0
            float* tmp_03 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_03, 0, sizeof(float) * dim1 * dim2 * dim3);
            mean_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_03, param);
            mean_3d_ax2_uint8(dim1, dim2, dim3, tmp, tmp_03);
            free(tmp_03);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 1 && param_dim1 == 2) || (param_dim0 == 2 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);
            mean_4d_ax1_uint8(dim0, dim1, dim2, dim3, data, tmp_12, param);
            mean_3d_ax1_uint8(dim0, dim2, dim3, tmp, tmp_12);

            free(tmp_12);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 1 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 1)))
        {
            // reduce on axis1
            float* tmp_13 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_13, 0, sizeof(float) * dim0 * dim2 * dim3);
            mean_4d_ax1_uint8(dim0, dim1, dim2, dim3, data, tmp_13, param);
            mean_3d_ax2_uint8(dim0, dim2, dim3, tmp, tmp_13);

            free(tmp_13);
        }
        else if(param_dim2 == -2 && param_dim3 == -2 &&
                ((param_dim0 == 2 && param_dim1 == 3) || (param_dim0 == 3 && param_dim1 == 2)))
        {
            // reduce on axis2
            float* tmp_23 = ( float* )malloc(sizeof(float) * dim0 * dim1 * dim3);
            memset(tmp_23, 0, sizeof(float) * dim0 * dim1 * dim3);
            mean_4d_ax2_uint8(dim0, dim1, dim2, dim3, data, tmp_23, param);
            mean_3d_ax2_uint8(dim0, dim1, dim3, tmp, tmp_23);

            free(tmp_23);
        }
        else if(param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 2) ||
                                     (param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 1) ||
                                     (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 2) ||
                                     (param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 0) ||
                                     (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 1) ||
                                     (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_0 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_0, 0, sizeof(float) * dim2 * dim3);

            mean_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_01, param);
            mean_3d_ax0_uint8(dim1, dim2, dim3, tmp_0, tmp_01);
            mean_2d_ax0_uint8(dim2, dim3, tmp, tmp_0);

            free(tmp_01);
            free(tmp_0);
        }
        else if(param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 1 && param_dim2 == 3) ||
                                     (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 1) ||
                                     (param_dim0 == 1 && param_dim1 == 0 && param_dim2 == 3) ||
                                     (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 0) ||
                                     (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 1) ||
                                     (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_01 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_01, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim2 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim2 * dim3);

            mean_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_01, param);
            mean_3d_ax0_uint8(dim1, dim2, dim3, tmp_1, tmp_01);
            mean_2d_ax1_uint8(dim2, dim3, tmp, tmp_1);

            free(tmp_01);
            free(tmp_1);
        }
        else if(param_dim3 == -2 && ((param_dim0 == 0 && param_dim1 == 2 && param_dim2 == 3) ||
                                     (param_dim0 == 0 && param_dim1 == 3 && param_dim2 == 2) ||
                                     (param_dim0 == 2 && param_dim1 == 0 && param_dim2 == 3) ||
                                     (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 0) ||
                                     (param_dim0 == 3 && param_dim1 == 0 && param_dim2 == 2) ||
                                     (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 0)))
        {
            // reduce on axis0
            float* tmp_02 = ( float* )malloc(sizeof(float) * dim1 * dim2 * dim3);
            memset(tmp_02, 0, sizeof(float) * dim1 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim1 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim1 * dim3);

            mean_4d_ax0_uint8(dim0, dim1, dim2, dim3, data, tmp_02, param);
            mean_3d_ax1_uint8(dim1, dim2, dim3, tmp_1, tmp_02);
            mean_2d_ax1_uint8(dim1, dim3, tmp, tmp_1);

            free(tmp_02);
            free(tmp_1);
        }
        else if(param_dim3 == -2 && ((param_dim0 == 1 && param_dim1 == 2 && param_dim2 == 3) ||
                                     (param_dim0 == 1 && param_dim1 == 3 && param_dim2 == 2) ||
                                     (param_dim0 == 2 && param_dim1 == 1 && param_dim2 == 3) ||
                                     (param_dim0 == 2 && param_dim1 == 3 && param_dim2 == 1) ||
                                     (param_dim0 == 3 && param_dim1 == 1 && param_dim2 == 2) ||
                                     (param_dim0 == 3 && param_dim1 == 2 && param_dim2 == 1)))
        {
            // reduce on axis0
            float* tmp_12 = ( float* )malloc(sizeof(float) * dim0 * dim2 * dim3);
            memset(tmp_12, 0, sizeof(float) * dim0 * dim2 * dim3);

            float* tmp_1 = ( float* )malloc(sizeof(float) * dim0 * dim3);
            memset(tmp_1, 0, sizeof(float) * dim0 * dim3);

            mean_4d_ax1_uint8(dim0, dim1, dim2, dim3, data, tmp_12, param);
            mean_3d_ax1_uint8(dim0, dim2, dim3, tmp_1, tmp_12);
            mean_2d_ax1_uint8(dim0, dim3, tmp, tmp_1);

            free(tmp_12);
            free(tmp_1);
        }
    }
    // pase to out_data
    for(int i = 0; i < out_size; i++)
    {
        out_data[i] = round(tmp[i] / param->scale[1]) + param->zero[1];
    }
    free(tmp);
    return 0;
}
// mean
void mean_4d_ax0_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param)
{
    for(int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        float s_tmp = 0.f;
        for(int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
            s_tmp += real_input0;
        }
        tmp[j] = s_tmp / dim0;
    }
}
void mean_4d_ax1_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param)
{
    for(int n = 0; n < dim0; n++)
    {
        for(int cw = 0; cw < dim2 * dim3; cw++)
        {
            float s_tmp = 0.f;
            for(int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
                s_tmp += real_input0;
            }
            tmp[n * dim2 * dim3 + cw] = s_tmp / dim1;
        }
    }
}
void mean_4d_ax2_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param)
{
    for(int n = 0; n < dim0; n++)
    {
        for(int h = 0; h < dim1; h++)
        {
            for(int c = 0; c < dim3; c++)
            {
                float s_tmp = 0.f;
                for(int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
                    s_tmp += real_input0;
                }
                tmp[n * dim1 * dim3 + h * dim3 + c] = s_tmp / dim2;
            }
        }
    }
}
void mean_4d_ax3_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param)
{
    for(int n = 0; n < dim0; n++)
    {
        for(int h = 0; h < dim1; h++)
        {
            for(int w = 0; w < dim2; w++)
            {
                float s_tmp = 0.f;
                for(int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
                    s_tmp += real_input0;
                }
                tmp[n * dim1 * dim2 + h * dim2 + w] = s_tmp / dim3;
            }
        }
    }
}
void mean_3d_ax0_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for(int wc = 0; wc < dim2 * dim3; wc++)
    {
        float s_tmp = 0.f;
        for(int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            s_tmp += tmp_01[index];
        }
        tmp[wc] = s_tmp / dim1;
    }
}
void mean_3d_ax1_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for(int h = 0; h < dim1; h++)
    {
        for(int c = 0; c < dim3; c++)
        {
            float s_tmp = 0.f;
            for(int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                s_tmp += tmp_02[index];
            }
            tmp[h * dim3 + c] = s_tmp / dim2;
        }
    }
}
void mean_3d_ax2_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for(int h = 0; h < dim1; h++)
    {
        for(int w = 0; w < dim2; w++)
        {
            float s_tmp = 0.f;
            for(int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                s_tmp += tmp_03[index];
            }
            tmp[h * dim2 + w] += s_tmp / dim3;
        }
    }
}
void mean_2d_ax0_uint8(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for(int w = 0; w < dim2; w++)
    {
        float s_tmp = 0.f;
        for(int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            s_tmp += tmp_0[index];
        }
        tmp[w] += s_tmp / dim1;
    }
}
void mean_2d_ax1_uint8(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for(int h = 0; h < dim1; h++)
    {
        float s_tmp = 0.f;
        for(int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            s_tmp += tmp_1[index];
        }
        tmp[h] += s_tmp / dim2;
    }
}

// sum
void sum_4d_ax0_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param)
{
    for(int j = 0; j < dim1 * dim2 * dim3; j++)
    {
        // nhwc
        for(int n = 0; n < dim0; n++)
        {
            int offset = n * dim1 * dim2 * dim3 + j;
            float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
            tmp[j] += real_input0;
        }
    }
}
void sum_4d_ax1_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param)
{
    for(int n = 0; n < dim0; n++)
    {
        for(int cw = 0; cw < dim2 * dim3; cw++)
        {
            for(int h = 0; h < dim1; h++)
            {
                int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + cw;
                float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
                tmp[n * dim2 * dim3 + cw] += real_input0;
            }
        }
    }
}
void sum_4d_ax2_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param)
{
    for(int n = 0; n < dim0; n++)
    {
        for(int h = 0; h < dim1; h++)
        {
            for(int c = 0; c < dim3; c++)
            {
                for(int w = 0; w < dim2; w++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
                    tmp[n * dim1 * dim3 + h * dim3 + c] += real_input0;
                }
            }
        }
    }
}
void sum_4d_ax3_uint8(int dim0, int dim1, int dim2, int dim3, uint8_t* data, float* tmp, reduce_param* param)
{
    for(int n = 0; n < dim0; n++)
    {
        for(int h = 0; h < dim1; h++)
        {
            for(int w = 0; w < dim2; w++)
            {
                for(int c = 0; c < dim3; c++)
                {
                    int offset = n * dim1 * dim2 * dim3 + h * dim2 * dim3 + w * dim3 + c;
                    float real_input0 = (data[offset] - param->zero[0]) * param->scale[0];
                    tmp[n * dim1 * dim2 + h * dim2 + w] += real_input0;
                }
            }
        }
    }
}
void sum_3d_ax0_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_01)
{
    for(int wc = 0; wc < dim2 * dim3; wc++)
    {
        for(int h = 0; h < dim1; h++)
        {
            int index = h * dim2 * dim3 + wc;
            tmp[wc] += tmp_01[index];
        }
    }
}
void sum_3d_ax1_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_02)
{
    for(int h = 0; h < dim1; h++)
    {
        for(int c = 0; c < dim3; c++)
        {
            for(int w = 0; w < dim2; w++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim3 + c] += tmp_02[index];
            }
        }
    }
}
void sum_3d_ax2_uint8(int dim1, int dim2, int dim3, float* tmp, float* tmp_03)
{
    for(int h = 0; h < dim1; h++)
    {
        for(int w = 0; w < dim2; w++)
        {
            for(int c = 0; c < dim3; c++)
            {
                int index = h * dim2 * dim3 + w * dim3 + c;
                tmp[h * dim2 + w] += tmp_03[index];
            }
        }
    }
}
void sum_2d_ax0_uint8(int dim1, int dim2, float* tmp, float* tmp_0)
{
    for(int w = 0; w < dim2; w++)
    {
        for(int h = 0; h < dim1; h++)
        {
            int index = h * dim2 + w;
            tmp[w] += tmp_0[index];
        }
    }
}
void sum_2d_ax1_uint8(int dim1, int dim2, float* tmp, float* tmp_1)
{
    for(int h = 0; h < dim1; h++)
    {
        for(int w = 0; w < dim2; w++)
        {
            int index = h * dim2 + w;
            tmp[h] += tmp_1[index];
        }
    }
}
