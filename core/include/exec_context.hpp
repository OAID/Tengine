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
 * Author: haitao@openailab.com
 */

#ifndef __EXEC_CONTEXT_HPP__
#define __EXEC_CONTEXT_HPP__

#include <string>
#include <vector>
#include <mutex>

namespace TEngine {

struct DevExecutor;

class ExecContext
{
public:
    ExecContext(const std::string& name, int empty_context) : name_(name)
    {
        InitContext(empty_context);
    };
    ExecContext(std::string&& name, int empty_context) : name_(name)
    {
        InitContext(empty_context);
    };

    ExecContext(const ExecContext& src) = delete;
    ExecContext(ExecContext&& src) = delete;

    bool AddDevice(const char* dev_name);
    bool RemoveDevice(const char* dev_name);
    int GetDeviceNum(void);
    const char* GetDevice(int idx);
    bool ExistDevice(const char* dev_name);

    bool SetAttr(const char* attr_name, const void* val, int val_size);
    bool GetAttr(const char* attr_name, void* val, int val_size);

    static ExecContext* GetDefaultContext(void);

private:
    void InitContext(int empty_context);
    std::vector<DevExecutor*>::iterator FindDevice(const char* dev_name);

    std::string name_;
    std::vector<DevExecutor*> dev_list_;
    std::mutex lock_;
};

}    // namespace TEngine

#endif
