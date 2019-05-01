static int eltwise_uint8(uint8_t* output, uint8_t* input0, uint8_t* input1,int type, int input_count4,
                        int input_chan,int input_chan_1,int input_hw, int input_hw_1,int input1_count4,
                        int input_h,int input_w,int input_h_1,int input_w_1,int input_n,
                        int input_n_1,int layout,int out_size,float * output_buf,eltwise_param* param)
{
    switch(type)
    {
         case 10:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = ((*input0++) - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[0]- param->zero[1]) * param->scale[1];
                    float result=real_input0/real_input1;
                    *output++ =round(result / param->scale[2]) + param->zero[2];
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = (input0[i]- param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[i]- param->zero[1]) * param->scale[1];
                    float result=real_input0/real_input1;
                    *output++ = round(result / param->scale[2]) + param->zero[2];
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    float real_input0 = (input0[0]- param->zero[0]) * param->scale[0];
                    float real_input1 = ((*input1++)- param->zero[1]) * param->scale[1];
                    float result=real_input0/real_input1;
                    *output++ = round(result / param->scale[2]) + param->zero[2];
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
                                float real_input0 = (input0[ofset]- param->zero[0]) * param->scale[0];
                                float real_input1 = (input1[k]- param->zero[1]) * param->scale[1];
                                float result=real_input0/real_input1;
                                output[ofset] = round(result / param->scale[2]) + param->zero[2];
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
                                    float real_input0 = (input0[ofst]- param->zero[0]) * param->scale[0];
                                    float real_input1 = (input1[c]- param->zero[1]) * param->scale[1];
                                    float result=real_input0/real_input1;
                                    output[ofst]=round(result / param->scale[2]) + param->zero[2];
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
                                float real_input0 = (input0[k]- param->zero[0]) * param->scale[0];
                                float real_input1 = (input1[ofset]- param->zero[1]) * param->scale[1];
                                float result=real_input0/real_input1;
                                output[ofset] = round(result / param->scale[2]) + param->zero[2];
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
                                    float real_input0 = (input0[c]- param->zero[0]) * param->scale[0];
                                    float real_input1 = (input1[ofst]- param->zero[1]) * param->scale[1];
                                    float result=real_input0/real_input1;
                                    output[ofst]=round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = ((*input0++) - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[0]- param->zero[1]) * param->scale[1];
                    float result=real_input0*real_input1;
                    *output++ =round(result / param->scale[2]) + param->zero[2];
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = (input0[i]- param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[i]- param->zero[1]) * param->scale[1];
                    float result=real_input0*real_input1;
                    *output++ = round(result / param->scale[2]) + param->zero[2];
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    float real_input0 = (input0[0]- param->zero[0]) * param->scale[0];
                    float real_input1 = ((*input1++)- param->zero[1]) * param->scale[1];
                    float result=real_input0*real_input1;
                    *output++ = round(result / param->scale[2]) + param->zero[2];
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
                                float real_input0 = (input0[ofset]- param->zero[0]) * param->scale[0];
                                float real_input1 = (input1[k]- param->zero[1]) * param->scale[1];
                                float result=real_input0*real_input1;
                                output[ofset] = round(result / param->scale[2]) + param->zero[2];
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
                                    float real_input0 = (input0[ofst]- param->zero[0]) * param->scale[0];
                                    float real_input1 = (input1[c]- param->zero[1]) * param->scale[1];
                                    float result=real_input0*real_input1;
                                    output[ofst]=round(result / param->scale[2]) + param->zero[2];
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
                                float real_input0 = (input0[k]- param->zero[0]) * param->scale[0];
                                float real_input1 = (input1[ofset]- param->zero[1]) * param->scale[1];
                                float result=real_input0*real_input1;
                                output[ofset] = round(result / param->scale[2]) + param->zero[2];
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
                                    float real_input0 = (input0[c]- param->zero[0]) * param->scale[0];
                                    float real_input1 = (input1[ofst]- param->zero[1]) * param->scale[1];
                                    float result=real_input0*real_input1;
                                    output[ofst]=round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = ((*input0++) - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[0]- param->zero[1]) * param->scale[1];
                    float result=real_input0-real_input1;
                    *output++ =round(result / param->scale[2]) + param->zero[2];
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {   

                    float real_input0 = (input0[i]) * param->scale[0];
                    float real_input1 = (input1[i]) * param->scale[1];
                    float result=real_input0-real_input1;
                    *output = round(result / param->scale[2]);
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    float real_input0 = (input0[0]- param->zero[0]) * param->scale[0];
                    float real_input1 = ((*input1++)- param->zero[1]) * param->scale[1];
                    float result=real_input0-real_input1;
                    *output++ = round(result / param->scale[2]) + param->zero[2];
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
                                float real_input0 = (input0[ofset]- param->zero[0]) * param->scale[0];
                                float real_input1 = (input1[k]- param->zero[1]) * param->scale[1];
                                float result=real_input0-real_input1;
                                output[ofset] = round(result / param->scale[2]) + param->zero[2];
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
                                    float real_input0 = (input0[ofst]- param->zero[0]) * param->scale[0];
                                    float real_input1 = (input1[c]- param->zero[1]) * param->scale[1];
                                    float result=real_input0-real_input1;
                                    output[ofst]=round(result / param->scale[2]) + param->zero[2];
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
                                float real_input0 = (input0[k]- param->zero[0]) * param->scale[0];
                                float real_input1 = (input1[ofset]- param->zero[1]) * param->scale[1];
                                float result=real_input0-real_input1;
                                output[ofset] = round(result / param->scale[2]) + param->zero[2];
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
                                    float real_input0 = (input0[c]- param->zero[0]) * param->scale[0];
                                    float real_input1 = (input1[ofst]- param->zero[1]) * param->scale[1];
                                    float result=real_input0-real_input1;
                                    output[ofst]=round(result / param->scale[2]) + param->zero[2];
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
                    float real_input0 = ((*input0++) - param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[0]- param->zero[1]) * param->scale[1];
                    float result=real_input0+real_input1;
                    *output++ =round(result / param->scale[2]) + param->zero[2];
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    float real_input0 = (input0[i]- param->zero[0]) * param->scale[0];
                    float real_input1 = (input1[i]- param->zero[1]) * param->scale[1];
                    float result=real_input0+real_input1;
                    *output++ = round(result / param->scale[2]) + param->zero[2];
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    float real_input0 = (input0[0]- param->zero[0]) * param->scale[0];
                    float real_input1 = ((*input1++)- param->zero[1]) * param->scale[1];
                    float result=real_input0+real_input1;
                    *output++ = round(result / param->scale[2]) + param->zero[2];
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
                                float real_input0 = (input0[ofset]- param->zero[0]) * param->scale[0];
                                float real_input1 = (input1[k]- param->zero[1]) * param->scale[1];
                                float result=real_input0+real_input1;
                                output[ofset] = round(result / param->scale[2]) + param->zero[2];
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
                                    float real_input0 = (input0[ofst]- param->zero[0]) * param->scale[0];
                                    float real_input1 = (input1[c]- param->zero[1]) * param->scale[1];
                                    float result=real_input0+real_input1;
                                    output[ofst]=round(result / param->scale[2]) + param->zero[2];
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
                                float real_input0 = (input0[k]- param->zero[0]) * param->scale[0];
                                float real_input1 = (input1[ofset]- param->zero[1]) * param->scale[1];
                                float result=real_input0+real_input1;
                                output[ofset] = round(result / param->scale[2]) + param->zero[2];
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
                                    float real_input0 = (input0[c]- param->zero[0]) * param->scale[0];
                                    float real_input1 = (input1[ofst]- param->zero[1]) * param->scale[1];
                                    float result=real_input0+real_input1;
                                    output[ofst]=round(result / param->scale[2]) + param->zero[2];
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
                float real_input0 = (input0[i]- param->zero[0]) * param->scale[0];
                float result=exp(real_input0);
                *output++ = round(result / param->scale[2]) + param->zero[2];
            }
            break;
        }
        default:
            break;
    }
    return 0;
}