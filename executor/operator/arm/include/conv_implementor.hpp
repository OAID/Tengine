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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __CONV_IMPLEMENTOR_HPP__
#define __CONV_IMPLEMENTOR_HPP__

#include <string>
#include <cstdlib>

#include "op_implementor.hpp"

namespace TEngine {

struct  ConvImplementor: public OpImplementor<ConvParam *, Tensor *, Tensor *> {

    virtual bool Prerun(Node * node, ExecEngine * engine)=0;
    virtual bool Run(Node * node, ExecEngine * engine)=0;
    virtual bool Postrun(Node * node, ExecEngine * engine)=0;
    virtual bool Support(ConvParam * param, Tensor * input_tensor, Tensor * weight_tensor)=0;

    virtual ~ ConvImplementor() {}

    bool GetDefaultEngine(const char * engine_name)
    {
        static std::string def_engine="CONV_FAST";

        const char * str=std::getenv("CONV_DEF");

        if(str)
           def_engine=str;

        return def_engine==engine_name;
    }

};


class ConvImplementorManager: public ImplementorManager<ConvImplementor,ConvParam *, Tensor *, Tensor *> {};


} //namespace TEngine



#endif
