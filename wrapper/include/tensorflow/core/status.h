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
#ifndef __TENSORFLOW_CORE_STATUS_H__
#define __TENSORFLOW_CORE_STATUS_H__

#include <memory>

namespace tensorflow {

class Status
{
public:
    Status() {}

    Status(int code, std::string msg);

    bool ok() const
    {
        return (state_ == NULL);
    }

    int code() const
    {
        return ok() ? 0 : state_->code;
    }

    const std::string& error_message() const
    {
        return ok() ? empty_string() : state_->msg;
    }

private:
    static const std::string& empty_string();

    struct State
    {
        int code;
        std::string msg;
    };
    // OK status has a `NULL` state_.  Otherwise, `state_` points to
    // a `State` structure containing the error code and message(s)
    std::unique_ptr<State> state_;
};

}    // namespace tensorflow

#endif    // __TENSORFLOW_CORE_STATUS_H__
