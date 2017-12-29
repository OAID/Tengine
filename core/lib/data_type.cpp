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
#include <data_type.hpp>

#include <cstdlib>

namespace TEngine {


template <>
int DataType::Convert<int>(const std::string& str) const
{

    int base=10;

    if(str.substr(0,2)==std::string("0x"))
          base=16;

    return strtoul(str.c_str(),NULL,base);
}


template <>
float DataType::Convert<float>(const std::string& str) const
{
    return strtof(str.c_str(),NULL);
}


template <>
void NamedData<DataType>::InitPredefinedData()
{
    /* A few pre-defined data types */
    static DataType float32("float32",4,true);
    static DataType float16("float16",2);
    static DataType int32("int",4);
    static DataType int8("int8",1);
}

} //namespace TEngine
