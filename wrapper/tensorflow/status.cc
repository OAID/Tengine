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
#include <cassert>

#include "tensorflow/core/status.h"

namespace tensorflow {

Status::Status(int code, std::string message)
{
    assert(code != 0);
    state_ = std::unique_ptr<State>(new State);
    state_->code = code;
    state_->msg = message;
}

const std::string& Status::empty_string()
{
    static std::string* empty = new std::string;
    return *empty;
}

}  // namespace tensorflow
