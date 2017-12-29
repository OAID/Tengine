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
#ifndef __AUTO_REGISTER_HPP__
#define __AUTO_REGISTER_HPP__

#define UNIQ_DUMMY_NAME_WITH_LINE0(a,b) auto_dummy_##a##_##b

#define UNIQ_DUMMY_NAME_WITH_LINE(a,b) UNIQ_DUMMY_NAME_WITH_LINE0(a,b)

#define UNIQ_DUMMY_NAME(name)  UNIQ_DUMMY_NAME_WITH_LINE(name,__LINE__)


#define DUMMY_AUTO_FUNCTION(func, ...) \
        DUMMY_AUTO_REGISTER(func,func, __VA_ARGS__)


#define DUMMY_AUTO_REGISTER(name,func, ...) \
        namespace {\
           namespace UNIQ_DUMMY_NAME(name) { \
               struct dummy {  dummy() { func(__VA_ARGS__); }};\
               static dummy dummy_obj;\
           }\
       }

#endif

