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
 * Author: jingyou@openailab.com
 */
#ifndef __CAFFE_COMMON_HPP__
#define __CAFFE_COMMON_HPP__

#include <glog/logging.h>
#include <iostream>

#define Caffe Caffe_wrap

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)     \
    char gInstantiationGuard##classname; \
    template class classname<float>;     \
    template class classname<double>

namespace caffe {

class Caffe
{
public:
    enum Brew
    {
        CPU,
        GPU
    };

    static Caffe& Get();
    static void set_mode(Brew mode)
    {
        Get().mode_ = mode;
    }

protected:
    Brew mode_;
};

}    // namespace caffe

#endif    // __CAFFE_COMMON_HPP__
