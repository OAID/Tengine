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
#ifndef __REGION_HPP__
#define __REGION_HPP__

#include "operator.hpp"
#include "region_param.hpp"
namespace TEngine {

class Region: public OperatorWithParam<Region, RegionParam> {

public:

      Region() { name_="Region";}
      Region(const Region& src)=default;
      virtual ~Region() {};
      void SetSchema(void) override;
 
};

} //namespace TEngine



#endif
