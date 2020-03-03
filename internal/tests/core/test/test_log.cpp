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

#include "logger.hpp"

using namespace TEngine;

int main(void)
{
    LOG_INFO() << "hello, world\n";

    LogOption opt = GET_LOG_OPTION();

    opt.log_date = true;

    SET_LOG_OPTION(opt);

    LOG_INFO() << "Test again\n";

    SET_LOG_LEVEL(kAlert);

    LOG_ERROR() << "I should not be displayed" << std::endl;

    LOG_FATAL() << "This is a fatal message\n";

    opt.log_level = false;
    SET_LOG_OPTION(opt);

    LOG_ALERT() << "do not ignore me!!!\n";

    XLOG_FATAL() << " bye, world\n";

    return 0;
}
