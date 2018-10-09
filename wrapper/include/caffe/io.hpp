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
#ifndef __CAFFE_IO_H__
#define __CAFFE_IO_H__

#include <fstream>
#include <iomanip>
#include <iostream>
#include "google/protobuf/message.h"

#define ReadProtoFromBinaryFile ReadProtoFromBinaryFile_wrap
#define ReadProtoFromBinaryFileOrDie ReadProtoFromBinaryFileOrDie_wrap

namespace caffe {

using ::google::protobuf::Message;

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  if (!ReadProtoFromBinaryFile(filename, proto))
    std::cerr << "Parse file: " << filename << " failed\n";
}

}  // namespace caffe

#endif  // __CAFFE_IO_H__
