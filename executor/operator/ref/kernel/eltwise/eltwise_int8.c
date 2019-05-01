static int eltwise_int8(int8_t* output, int8_t* input0, int8_t* input1,int type, int input_count4,
                        int input_chan,int input_chan_1,int input_hw, int input_hw_1,int input1_count4,
                        int input_h,int input_w,int input_h_1,int input_w_1,int input_n,
                        int input_n_1,int layout,int out_size,float * output_buf,eltwise_param* param)
{

    // float * output_buf=(float *)malloc(sizeof(float)*out_size);
    switch(type)
    {
         case 10:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = ((*input0++) ) * param->scale[0];
                    float real_input1 = (input1[0]) * param->scale[1];
                    float result=real_input0/real_input1;
                    output_buf[i] =result;
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = (input0[i]) * param->scale[0];
                    float real_input1 = (input1[i]) * param->scale[1];
                    float result=real_input0/real_input1;
                    output_buf[i] = result;
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    float real_input0 = (input0[0]) * param->scale[0];
                    float real_input1 = ((*input1++)) * param->scale[1];
                    float result=real_input0/real_input1;
                    output_buf[i] = result;
                }
            }
            else if(input_chan == input1_count4)
            {
                if(layout==0)
                {
                    for(int j=0;j<input_n;j++)
                    {
                        for(int k=0;k<input_chan;k++)
                        {
                            for(int i = 0; i < input_hw; ++i)
                            {
                                int ofset=j*input_chan*input_hw+k*input_hw+i;
                                float real_input0 = (input0[ofset]) * param->scale[0];
                                float real_input1 = (input1[k]) * param->scale[1];
                                float result=real_input0/real_input1;
                                output_buf[ofset] = result;
                            }
                        }
                    }
                }
                //nhwc
                else
                {
                    for(int i=0;i<input_n;i++)
                    {
                        for(int j=0;j<input_h;j++)
                        {
                            for(int k=0;k<input_w;k++)
                            {
                                for(int c=0;c<input_chan;c++)
                                {
                                    int ofst=i*input_h*input_w*input_chan
                                    +j*input_w*input_chan+k*input_chan+c;
                                    float real_input0 = (input0[ofst]) * param->scale[0];
                                    float real_input1 = (input1[c]) * param->scale[1];
                                    float result=real_input0/real_input1;
                                    output_buf[ofst]=result;
                                }
                            }
                        }
                    }
                }
            }
            else if(input_chan_1 == input_count4)
            {
                if(layout==0)
                {
                    for(int j=0;j<input_n_1;j++)
                    {
                        for(int k=0;k<input_chan_1;k++)
                        {
                            for(int i = 0; i < input_hw_1; ++i)
                            {
                                int ofset=j*input_chan_1*input_hw_1+k*input_hw_1+i;
                                float real_input0 = (input0[k]) * param->scale[0];
                                float real_input1 = (input1[ofset]) * param->scale[1];
                                float result=real_input0/real_input1;
                                output_buf[ofset] = result;
                            }
                        }
                    }
                }
                else
                {
                    for(int i=0;i<input_n_1;i++)
                    {
                        for(int j=0;j<input_h_1;j++)
                        {
                            for(int k=0;k<input_w_1;k++)
                            {
                                for(int c=0;c<input_chan_1;c++)
                                {
                                    int ofst=i*input_h_1*input_w_1*input_chan_1
                                    +j*input_w_1*input_chan_1+k*input_chan_1+c;
                                    float real_input0 = (input0[c]) * param->scale[0];
                                    float real_input1 = (input1[ofst]) * param->scale[1];
                                    float result=real_input0/real_input1;
                                    output_buf[ofst]=result;
                                }
                            }
                        }
                    }
                }
                
            }
            else
            {
                return -1;
            }
            break;
        }
        case 0:
        { 
           if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = ((*input0++) ) * param->scale[0];
                    float real_input1 = (input1[0]) * param->scale[1];
                    float result=real_input0*real_input1;
                    printf("result: %f\n",result);
                    output_buf[i] =result;
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = (input0[i]) * param->scale[0];
                    float real_input1 = (input1[i]) * param->scale[1];
                    float result=real_input0*real_input1;
                    output_buf[i] = result;
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    float real_input0 = (input0[0]) * param->scale[0];
                    float real_input1 = ((*input1++)) * param->scale[1];
                    float result=real_input0*real_input1;
                    output_buf[i] = result;
                }
            }
            else if(input_chan == input1_count4)
            {
                
                if(layout==0)
                {
                   for(int j=0;j<input_n;j++)
                    {
                        for(int k=0;k<input_chan;k++)
                        {
                            for(int i = 0; i < input_hw; ++i)
                            {
                                int ofset=j*input_chan*input_hw+k*input_hw+i;
                                float real_input0 = (input0[ofset]) * param->scale[0];
                                float real_input1 = (input1[k]) * param->scale[1];
                                float result=real_input0*real_input1;
                                output_buf[ofset] = result;
                            }
                        }
                    }
                }
                //nhwc
                else
                {
                    for(int i=0;i<input_n;i++)
                    {
                        for(int j=0;j<input_h;j++)
                        {
                            for(int k=0;k<input_w;k++)
                            {
                                for(int c=0;c<input_chan;c++)
                                {
                                    int ofst=i*input_h*input_w*input_chan
                                    +j*input_w*input_chan+k*input_chan+c;
                                    float real_input0 = (input0[ofst]) * param->scale[0];
                                    float real_input1 = (input1[c]) * param->scale[1];
                                    float result=real_input0*real_input1;
                                    output_buf[ofst]=result;
                                }
                            }
                        }
                    }
                }
            }
            else if(input_chan_1 == input_count4)
            {
                if(layout==0)
                {
                    for(int j=0;j<input_n_1;j++)
                    {
                        for(int k=0;k<input_chan_1;k++)
                        {
                            for(int i = 0; i < input_hw_1; ++i)
                            {
                                int ofset=j*input_chan_1*input_hw_1+k*input_hw_1+i;
                                float real_input0 = (input0[k]) * param->scale[0];
                                float real_input1 = (input1[ofset]) * param->scale[1];
                                float result=real_input0*real_input1;
                                output_buf[ofset] = result;
                            }
                        }
                    }
                }
                else
                {
                    for(int i=0;i<input_n_1;i++)
                    {
                        for(int j=0;j<input_h_1;j++)
                        {
                            for(int k=0;k<input_w_1;k++)
                            {
                                for(int c=0;c<input_chan_1;c++)
                                {
                                    int ofst=i*input_h_1*input_w_1*input_chan_1
                                    +j*input_w_1*input_chan_1+k*input_chan_1+c;
                                    float real_input0 = (input0[c]) * param->scale[0];
                                    float real_input1 = (input1[ofst]) * param->scale[1];
                                    float result=real_input0*real_input1;
                                    output_buf[ofst]=result;
                                }
                            }
                        }
                    }
                }
                
            }
            else
            {
                return -1;
            }
            break;
        }
        case 4:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = ((*input0++) ) * param->scale[0];
                    float real_input1 = (input1[0]) * param->scale[1];
                    float result=real_input0-real_input1;
                    output_buf[i] =result;
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {   

                    float real_input0 = (input0[i]) * param->scale[0];
                    float real_input1 = (input1[i]) * param->scale[1];
                    float result=real_input0-real_input1;
                    *output_buf = result;
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    float real_input0 = (input0[0]) * param->scale[0];
                    float real_input1 = ((*input1++)) * param->scale[1];
                    float result=real_input0-real_input1;
                    output_buf[i] = result;
                }
            }
            else if(input_chan == input1_count4)
            {
                if(layout==0)
                {
                   for(int j=0;j<input_n;j++)
                    {
                        for(int k=0;k<input_chan;k++)
                        {
                            for(int i = 0; i < input_hw; ++i)
                            {
                                int ofset=j*input_chan*input_hw+k*input_hw+i;
                                float real_input0 = (input0[ofset]) * param->scale[0];
                                float real_input1 = (input1[k]) * param->scale[1];
                                float result=real_input0-real_input1;
                                output_buf[ofset] = result;
                            }
                        }
                    }
                }
                //nhwc
                else
                {
                    for(int i=0;i<input_n;i++)
                    {
                        for(int j=0;j<input_h;j++)
                        {
                            for(int k=0;k<input_w;k++)
                            {
                                for(int c=0;c<input_chan;c++)
                                {
                                    int ofst=i*input_h*input_w*input_chan
                                    +j*input_w*input_chan+k*input_chan+c;
                                    float real_input0 = (input0[ofst]) * param->scale[0];
                                    float real_input1 = (input1[c]) * param->scale[1];
                                    float result=real_input0-real_input1;
                                    output_buf[ofst]=result;
                                }
                            }
                        }
                    }
                }
            }
            else if(input_chan_1 == input_count4)
            {
                if(layout==0)
                {
                    for(int j=0;j<input_n_1;j++)
                    {
                        for(int k=0;k<input_chan_1;k++)
                        {
                            for(int i = 0; i < input_hw_1; ++i)
                            {
                                int ofset=j*input_chan_1*input_hw_1+k*input_hw_1+i;
                                float real_input0 = (input0[k]) * param->scale[0];
                                float real_input1 = (input1[ofset]) * param->scale[1];
                                float result=real_input0-real_input1;
                                output_buf[ofset] = result;
                            }
                        }
                    }
                }
                else
                {
                    for(int i=0;i<input_n_1;i++)
                    {
                        for(int j=0;j<input_h_1;j++)
                        {
                            for(int k=0;k<input_w_1;k++)
                            {
                                for(int c=0;c<input_chan_1;c++)
                                {
                                    int ofst=i*input_h_1*input_w_1*input_chan_1
                                    +j*input_w_1*input_chan_1+k*input_chan_1+c;
                                    float real_input0 = (input0[c]) * param->scale[0];
                                    float real_input1 = (input1[ofst]) * param->scale[1];
                                    float result=real_input0-real_input1;
                                    output_buf[ofst]=result;
                                }
                            }
                        }
                    }
                }
                
            }
            else
            {
                return -1;
            }
            break;
        }
        case 2:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = ((*input0++) ) * param->scale[0];
                    float real_input1 = (input1[0]) * param->scale[1];
                    float result=real_input0+real_input1;
                    output_buf[i] =result;
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = (input0[i]) * param->scale[0];
                    float real_input1 = (input1[i]) * param->scale[1];
                    float result=real_input0+real_input1;
                    output_buf[i] = result;
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    float real_input0 = (input0[0]) * param->scale[0];
                    float real_input1 = ((*input1++)) * param->scale[1];
                    float result=real_input0+real_input1;
                    output_buf[i] = result;
                }
            }
            else if(input_chan == input1_count4)
            {
                if(layout==0)
                {
                   for(int j=0;j<input_n;j++)
                    {
                        for(int k=0;k<input_chan;k++)
                        {
                            for(int i = 0; i < input_hw; ++i)
                            {
                                int ofset=j*input_chan*input_hw+k*input_hw+i;
                                float real_input0 = (input0[ofset]) * param->scale[0];
                                float real_input1 = (input1[k]) * param->scale[1];
                                float result=real_input0+real_input1;
                                output_buf[ofset] = result;
                            }
                        }
                    }
                }
                //nhwc
                else
                {
                    for(int i=0;i<input_n;i++)
                    {
                        for(int j=0;j<input_h;j++)
                        {
                            for(int k=0;k<input_w;k++)
                            {
                                for(int c=0;c<input_chan;c++)
                                {
                                    int ofst=i*input_h*input_w*input_chan
                                    +j*input_w*input_chan+k*input_chan+c;
                                    float real_input0 = (input0[ofst]) * param->scale[0];
                                    float real_input1 = (input1[c]) * param->scale[1];
                                    float result=real_input0+real_input1;
                                    output_buf[ofst]=result;
                                }
                            }
                        }
                    }
                }
            }
            else if(input_chan_1 == input_count4)
            {
                if(layout==0)
                {
                    for(int j=0;j<input_n_1;j++)
                    {
                        for(int k=0;k<input_chan_1;k++)
                        {
                            for(int i = 0; i < input_hw_1; ++i)
                            {
                                int ofset=j*input_chan_1*input_hw_1+k*input_hw_1+i;
                                float real_input0 = (input0[k]) * param->scale[0];
                                float real_input1 = (input1[ofset]) * param->scale[1];
                                float result=real_input0+real_input1;
                                output_buf[ofset] = result;
                            }
                        }
                    }
                }
                else
                {
                    for(int i=0;i<input_n_1;i++)
                    {
                        for(int j=0;j<input_h_1;j++)
                        {
                            for(int k=0;k<input_w_1;k++)
                            {
                                for(int c=0;c<input_chan_1;c++)
                                {
                                    int ofst=i*input_h_1*input_w_1*input_chan_1
                                    +j*input_w_1*input_chan_1+k*input_chan_1+c;
                                    float real_input0 = (input0[c]) * param->scale[0];
                                    float real_input1 = (input1[ofst]) * param->scale[1];
                                    float result=real_input0+real_input1;
                                    output_buf[ofst]=result;
                                }
                            }
                        }
                    }
                }
                
            }
            else
            {
                return -1;
            }
            break;
        }
        case 12:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                float real_input0 = (input0[i]) * param->scale[0];
                float result=exp(real_input0);
                output_buf[i] = result;
            }
            break;
        }
        default:
            break;
    }
    float output_max = 0.0f;
    for(int i =0; i< out_size; i++)
    {
        if(output_max < fabs(output_buf[i]))
            output_max = fabs(output_buf[i]);
    }
    param->scale[2] = output_max/127;
    for(int i =0; i< out_size; i++)
    {
        output[i] = round(output_buf[i]*127/output_max);
    }

    return 0;
}