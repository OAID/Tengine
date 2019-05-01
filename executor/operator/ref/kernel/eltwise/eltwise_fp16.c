static int eltwise_fp16(__fp16* output, __fp16* input0, __fp16* input1,int type, int input_count4,
                        int input_chan,int input_chan_1,int input_hw, int input_hw_1,int input1_count4,
                        int input_h,int input_w,int input_h_1,int input_w_1,int input_n,
                        int input_n_1,int layout,int out_size,float * output_buf,eltwise_param* param)
{
#if!defined( __ARM_ARCH) || __ARM_ARCH <8
    switch(type)
    {
        case 10:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32((*input0++)) / fp16_to_fp32(input1[0]));
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32(input0[i]) / fp16_to_fp32(input1[i]));
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32(input0[0])/fp16_to_fp32((*input1++)));
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
                                output[ofset] = fp32_to_fp16(fp16_to_fp32(input0[ofset]) / fp16_to_fp32(input1[k]));
                            }
                        }
                    }
                }
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
                                    output[ofst]=fp32_to_fp16(fp16_to_fp32(input0[ofst]) / fp16_to_fp32(input1[c]));
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
                                output[ofset] = fp32_to_fp16(fp16_to_fp32(input0[k]) / fp16_to_fp32(input1[ofset]));
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
                                    output[ofst]=fp32_to_fp16(fp16_to_fp32(input0[c]) / fp16_to_fp32(input1[ofst]));
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
                    *output++ = fp32_to_fp16(fp16_to_fp32((*input0++)) * fp16_to_fp32(input1[0]));
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32(input0[i]) * fp16_to_fp32(input1[i]));
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32((*input1++)) * fp16_to_fp32(input0[0]));
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
                                output[ofset] = fp32_to_fp16(fp16_to_fp32(input0[ofset]) * fp16_to_fp32(input1[k]));
                            }
                        }
                    }
                }
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

                                    output[ofst]=fp32_to_fp16(fp16_to_fp32(input0[ofst]) * fp16_to_fp32(input1[c]));
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
                                output[ofset] = fp32_to_fp16(fp16_to_fp32(input0[k]) * fp16_to_fp32(input1[ofset]));
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

                                    output[ofst]=fp32_to_fp16(fp16_to_fp32(input0[c]) * fp16_to_fp32(input1[ofst]));
                                }
                            }
                        }
                    }
                }
                
                
            }
            else
                return -1;
            break;
        }
        case 4:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32((*input0++)) - fp16_to_fp32(input1[0]));
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    output[i] = fp32_to_fp16(fp16_to_fp32(input0[i])-fp16_to_fp32(input1[i]));
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32(input0[0]) - fp16_to_fp32((*input1++)));
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
                                output[ofset] = fp32_to_fp16(fp16_to_fp32(input0[ofset]) - fp16_to_fp32(input1[k]));
                            }
                        }
                    }
                }
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

                                    output[ofst]=fp32_to_fp16(fp16_to_fp32(input0[ofst]) - fp16_to_fp32(input1[c]));
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
                                output[ofset] = fp32_to_fp16(fp16_to_fp32(input0[k]) - fp16_to_fp32(input1[ofset]));
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

                                    output[ofst]=fp32_to_fp16(fp16_to_fp32(input0[c]) - fp16_to_fp32(input1[ofst]));
                                }
                            }
                        }
                    }
                }
                
            }
            else
                return -1;
            break;
        }
        case 2:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32((*input0++)) + fp16_to_fp32(input1[0]));
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32((*input0++))+ fp16_to_fp32((*input1++)));
                }
            }
            else if (input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = fp32_to_fp16(fp16_to_fp32((*input1++)) + fp16_to_fp32(input0[0]));
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
                                output[ofset] = fp32_to_fp16(fp16_to_fp32(input0[ofset]) + fp16_to_fp32(input1[k]));
                            }
                        }
                    }
                }
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

                                    output[ofst]=fp32_to_fp16(fp16_to_fp32(input0[ofst]) + fp16_to_fp32(input1[c]));
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
                                output[ofset] = fp32_to_fp16(fp16_to_fp32(input0[k]) + fp16_to_fp32(input1[ofset]));
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

                                    output[ofst]=fp32_to_fp16(fp16_to_fp32(input0[c]) + fp16_to_fp32(input1[ofst]));
                                }
                            }
                        }
                    }
                }
                
            }
            else
                return -1;
            break;
        }
        case 11:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(log(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 12:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(exp(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 7:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(1 / sqrt(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 13:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(sqrt(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 14:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(floor(fp16_to_fp32(input0[i])));
            }
            break;
        }
        case 15:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = fp32_to_fp16(pow(fp16_to_fp32(input0[i]),2));
            }
            break;
        }
        // case 11:
        // {
        //     for(int i = 0; i < input_count4; ++i)
        //     {
        //         output[0] += input0[i];
        //     }
        //     break;
        // }
        // case 17:
        // {
        //     float sum=0;
        //     for(int i = 0; i < input_count4; ++i)
        //     {
        //         sum += input0[i];
        //     }
        //     output[0]=sum / input_count4;
        //     break;
        // }
        case 16:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = fp32_to_fp16(pow(fp16_to_fp32(input0[i]),fp16_to_fp32(input1[0])));
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = fp32_to_fp16(pow(fp16_to_fp32(input0[i]),fp16_to_fp32(input1[i])));
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = fp32_to_fp16(pow(fp16_to_fp32(input0[0]),fp16_to_fp32(input1[i])));
                }
            }
            else
            {
                return -1;
            }
            break;
        }
        default:
            break;
    }

#else
   switch(type)
    {
        case 10:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = (*input0++) / input1[0];
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = input0[i] / input1[i];
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = input0[0]/(*input1++);
                }
            }
            else if(input_chan == input1_count4)
            {
                //nchw
                if(layout==0)
                {
                    for(int j=0;j<input_n;j++)
                    {
                        for(int k=0;k<input_chan;k++)
                        {
                            for(int i = 0; i < input_hw; ++i)
                            {
                                int ofset=j*input_chan*input_hw+k*input_hw+i;
                                output[ofset] = input0[ofset] / input1[k];
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
                                    output[ofst]=input0[ofst] / input1[c];
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
                                output[ofset] = input0[k] / input1[ofset];
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

                                    output[ofst]=input0[c] / input1[ofst];
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
                    *output++ = (*input0++) * input1[0];
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = input0[i] * input1[i];
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = (*input1++) * input0[0];
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
                                output[ofset] = input0[ofset] * input1[k];
                                
                            }
                        }
                    }
                }
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

                                    output[ofst]=input0[ofst] * input1[c];
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
                                output[ofset] = input0[k] * input1[ofset];
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

                                    output[ofst]=input0[c] * input1[ofst];
                                }
                            }
                        }
                    }
                }
                
            }
            else
                return -1;
            break;
        }
        case 4:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = (*input0++) - input1[0];
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = (*input0++) - (*input1++);
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = input0[0] - (*input1++);
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
                                output[ofset] = input0[ofset] - input1[k];
                            }
                        }
                    }
                }
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

                                    output[ofst]=input0[ofst] - input1[c];
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
                                output[ofset] = input0[k] - input1[ofset];
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

                                    output[ofst]=input0[c] - input1[ofst];
                                }
                            }
                        }
                    }
                }
                
               
            }
            else
                return -1;
            break;
        }
        case 2:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = (*input0++) + input1[0];
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = (*input0++)+ (*input1++);
                }
            }
            else if (input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = (*input1++) + input0[0];
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

                                output[ofset] = input0[ofset] + input1[k];
                            }
                        }
                    }
                }
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

                                    output[ofst]=input0[ofst] + input1[c];
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
                                output[ofset] = input0[k] + input1[ofset];
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

                                    output[ofst]=input0[c] + input1[ofst];
                                }
                            }
                        }
                    }
                }
                
                
            }
            else
                return -1;
            break;
        }
        case 11:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = log(input0[i]);
            }
            break;
        }
        case 12:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = exp(input0[i]);
            }
            break;
        }
        case 7:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = 1 / sqrt(input0[i]);
            }
            break;
        }
        case 13:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = sqrt(input0[i]);
            }
            break;
        }
        case 14:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = floor(input0[i]);
            }
            break;
        }
        case 15:
        {
            for(int i = 0; i < input_count4; ++i)
            {
                *output++ = pow(input0[i],2);
            }
            break;
        }
        case 16:
        {
            if(input1_count4 == 1)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = pow(input0[i],input1[0]);
                }
            }
            else if(input_count4 == input1_count4)
            {
                for(int i = 0; i < input_count4; ++i)
                {
                    *output++ = pow(input0[i],input1[i]);
                }
            }
            else if(input_count4 == 1)
            {
                for(int i = 0; i < input1_count4; ++i)
                {
                    *output++ = pow(input0[0],input1[i]);
                }
            }
            else
            {
                return -1;
            }
            break;
        }
        default:
            break;
    }
    #endif
    return 0;
}