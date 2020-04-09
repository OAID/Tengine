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
 * Copyright (c) 2019, Open AI Lab
 * Author: jjzeng@openailab.com
 */

#ifndef __TENGINEREPORT_MGR_HPP__
#define __TENGINEREPORT_MGR_HPP__

#include "reportdata.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_tengine_report_mgr();
void release_tengine_report_mgr();
void do_tengine_report(int action);
int set_tengine_report_stat(int stat);

#ifdef __cplusplus
}
#endif

#endif
