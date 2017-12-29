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
#ifndef __CONST_OPERATOR_HPP__
#define __CONST_OPERATOR_HPP__


#include "operator.hpp"


namespace TEngine {

class ConstOp: public OperatorNoParam<ConstOp> {

public:
       ConstOp() {name_="Const";}
       ConstOp(const ConstOp& src)=default;
       virtual ~ ConstOp() {};

       void SetSchema(void) override;

}; 



} //namespace TEngine




#endif
