#ifndef __TENGINE_CONVOLUTION_OP_HPP__
#define __TENGINE_CONVOLUTION_OP_HPP__

#include <string>

#include "tengine_c_api.h"
#include "tengine_op.hpp"

namespace tengine
{
    namespace nn
    {
        class TengineConvolution : public TengineOp
        {
            public:
                ~TengineConvolution();
                virtual bool run();

                bool init(OpData& input,OpData& output,int group,float* kernel,float kernel_s,
                    int kernel_h,int kernel_w,float* teg_bias,int stride_h,int stride_w,int pad_h,int pad_w,
                    int dilation_h,int dilation_w,size_t wstep,const std::string& padMode);

            protected:
                TengineConvolution(){}

            private:
                graph_t _graph;

            public:
                static TTengineOpPtr create(OpData& input,OpData& output,int group,float* kernel,float kernel_s,
                    int kernel_h,int kernel_w,float* teg_bias,int stride_h,int stride_w,int pad_h,int pad_w,
                    int dilation_h,int dilation_w,size_t wstep,const std::string& padMode);
        };

    }

}

#endif
