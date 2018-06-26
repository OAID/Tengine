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
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#ifndef __ELTWISE_HPP__
#define __ELTWISE_HPP__

#include "operator.hpp"
#include "eltwise_param.hpp"
namespace TEngine {

class Eltwise: public OperatorWithParam<Eltwise, EltwiseParam> {

public:

      Eltwise() { name_="Eltwise";}
      Eltwise(const Eltwise& src)=default;
      virtual ~Eltwise() {};

     void MethodToType(EltwiseParam& param)
     {
         std::string& method=param.method;

         /* default eltwise_SUM */
         param.type=ELT_SUM;

         if(method == "max")
             param.type=ELT_MAX;
         else if(method =="prod")
            param.type=ELT_PROD;
     }
      void ParseParam(EltwiseParam & param, Operator * op) override
     {
         ParsePredefinedParam(param,op);
         MethodToType(param);
     }
      void SetSchema(void) override;
 
};

} //namespace TEngine



#endif
