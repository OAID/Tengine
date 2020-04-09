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
 * Copyright (c) 2020, Open AI Lab
 * Author: jjzeng@openailab.com
 */

#ifndef __OP_UTILS_HPP__ 
#define __OP_UTILS_HPP__

#define ON_A72 1
#define ON_A53 2 
#define ON_A17 3
#define ON_A7 4
#define ON_A55 5
#define ON_KRYO 6
#define ON_A73 7
#define ON_A9 8
#define ON_A15 9

#ifdef OFF_A72
#undef ON_A72
#endif

#ifdef OFF_A53
#undef ON_A53
#endif

#ifdef OFF_A17
#undef ON_A17
#endif

#ifdef OFF_A7
#undef ON_A7
#endif

#ifdef OFF_A55
#undef ON_A55
#endif

#ifdef OFF_A73
#undef ON_A73
#endif

#ifdef OFF_A9
#undef ON_A9
#endif

#ifdef OFF_A15
#undef ON_A15
#endif

#endif
