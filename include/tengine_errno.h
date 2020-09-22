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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: haitao@openailab.com
 */

#ifndef __TENGINE_ERRNO_H__
#define __TENGINE_ERRNO_H__

#include <errno.h>

#ifdef CONFIG_ARCH_CORTEX_M

#define ENOENT 102
#define EAGAIN 111
#define EFAULT 114
#define EEXIST 117
#define ENOSPC 128
#define ENODATA 161
#define ENOTSUP 195

#endif

extern int get_tengine_errno(void);
extern void set_tengine_errno(int err_num);

#endif
