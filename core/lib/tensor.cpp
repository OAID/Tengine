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
#include <iostream>
#include <string>

#include "data_type.hpp"
#include "tensor.hpp"

namespace TEngine {

void Tensor::Reshape(const TShape& shape)
{

   if(shape_==shape)
         return;

   shape_=shape;
}

unsigned int Tensor::GetTotalSize(void) const
{
    const DataType * dtype=DataType::GetType(data_type_);

    unsigned int elem_size=dtype->GetTypeSize();
    unsigned int elem_num=shape_.GetSize();

    return elem_size*elem_num;
}

static std::string MapTypeToString(TensorType type)
{
   if(type == kVarTensor)
       return "Var";
   else if(type==kConstTensor)
       return "Const";
   else if(type == kInputTensor)
       return "Input";
   else
       return "Unknown";
}


void  Tensor::DumpTensor(std::ostream& os) const
{
    os<<name_<<" type: "<<MapTypeToString(type_)<<"  Shape: ";
    shape_.DumpShape(os);
}



} //namespace TEngine

