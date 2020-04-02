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
 * Author: honggui@openailab.com
 */
#pragma once
#include <string>
#include <stdexcept>

namespace TEngine {
using error_code_t = std::string;
struct te_error_base : public std::runtime_error
{
    error_code_t error_code;
    virtual error_code_t get_error_code()
    {
        return error_code;
    }
<<<<<<< HEAD
    te_error_base() : runtime_error("tengine error"){}
=======
    te_error_base() : runtime_error("tengine error") {}
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
};
struct te_error_shared_function_not_found : public te_error_base
{
    using te_error_base::te_error_base;
    static std::string msg;
<<<<<<< HEAD
    
    te_error_shared_function_not_found(const std::string& func_name) 
    {
        msg="\nShared function not found: ";
        msg+=func_name;
        msg+="\n";
=======

    te_error_shared_function_not_found(const std::string& func_name)
    {
        msg = "\nShared function not found: ";
        msg += func_name;
        msg += "\n";
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
    }

    const char* what() const throw() override
    {
<<<<<<< HEAD
        return msg.c_str(); 
=======
        return msg.c_str();
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
    }
};
struct te_error_unable_to_load_library : public te_error_base
{
    using te_error_base::te_error_base;
    static std::string msg;

    te_error_unable_to_load_library(const std::string& so_name)
    {
<<<<<<< HEAD
        msg="\nShared library not found: ";
        msg+=so_name;
        msg+="\n";
    } 
=======
        msg = "\nShared library not found: ";
        msg += so_name;
        msg += "\n";
    }
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

    const char* what() const throw() override
    {
        return msg.c_str();
    }
};
struct te_error_general : public te_error_base
{
    using te_error_base::te_error_base;
    const char* what() const throw() override
    {
        return error_code.c_str();
    }
};
}    // namespace TEngine
