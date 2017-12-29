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


#include "tensor.hpp"

using namespace TEngine;



int main(void)
{

   /* test TShape */

   TShape shape;

   shape.SetDataLayout("NCHW");
   std::vector<int> dim={1,2,3,4};

   try
   {
        shape.SetDim(dim,true);
   }
   catch(std::exception &e)
   {
        std::cout<<"detected error"<<std::endl;
   }

   shape.SetDim(dim);

   std::cout<<shape.GetSize()<<std::endl; 

   std::cout<<"H is: "<<shape.GetH()<<std::endl;


   TShape shape2;

   shape2=shape;


   std::cout<<"W is: "<<shape2.GetW()<<std::endl;

   std::cout<< (shape == shape2) <<std::endl;

   dim[2]=10;

   shape2.SetDim(dim);

   std::cout<< (shape == shape2) <<std::endl;

   /* Tets Tensor */

   Tensor tensor("test");

   tensor.SetDataType("int");
   tensor.Reshape(shape);

   std::cout<<"ALL TESTS DONE"<<std::endl;

   return 0;
}
