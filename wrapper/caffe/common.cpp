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
#include "caffe/common.hpp"
#include <boost/thread.hpp>

namespace caffe {

// Make sure each thread can have different values.
static boost::thread_specific_ptr<Caffe> thread_instance_;

Caffe& Caffe::Get() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Caffe());
  }
  return *(thread_instance_.get());
}

}  // namespace caffe
