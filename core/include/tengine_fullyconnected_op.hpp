/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, Open AI Lab
 * Author: jjzeng@openailab.com
 */

#ifndef __TENGINE_FULLYCONNECTED_OP_HPP__
#define __TENGINE_FULLYCONNECTED_OP_HPP__

#include <string>

#include "tengine_c_api.h"
#include "tengine_op.hpp"

namespace tengine
{
    namespace nn
    {
        class TengineFullyConnected : public TengineOp
        {
            public:
                ~TengineFullyConnected();
                virtual bool run();
                virtual bool valid()const;
                virtual Tensor* get_output_tensor()const;

                bool init(Tensor& input,const char* weight,const char* bias,int num_output);

            protected:
                TengineFullyConnected(){}

            private:
                graph_t _graph;

            public:
                static TTengineOpPtr create(Tensor& input,const char* weight,const char* bias,int num_output);
        };

    }

}

#endif
