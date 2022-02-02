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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: haitao@openailab.com
 * Revised: lswang@openailab.com
 */

#pragma once

#include "api/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

struct log_option
{
    int print_prefix;
    int print_time;
    int print_level;
};

struct logger
{
    const char* prefix;
    int log_level;
    struct log_option option;

    void (*output_func)(const char*);

    void (*log)(struct logger*, enum log_level, const char* fmt, ...);
    void (*set_log_level)(struct logger*, int level);
    void (*set_output_func)(struct logger*, void (*func)(const char*));
};

struct logger* get_default_logger(void);

#define SET_LOG_OUTPUT(func)                          \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->set_output_func(logger, func);        \
    } while (0)

#define SET_LOG_LEVEL(level)                          \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->set_log_level(logger, level);         \
    } while (0)

#define SET_LOG_PRINT_TIME(val)                       \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->option.print_time = val;              \
    } while (0)

#define SET_LOG_PRINT_LEVEL(val)                      \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->option.print_level = val;             \
    } while (0)

#define SET_LOG_PRINT_PREFIX(val)                     \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->option.print_prefix = val;            \
    } while (0)

#define SET_LOG_PREFIX(val)                           \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->prefix = val;                         \
    } while (0)

#define LOG(level, fmt, ...)                            \
    do                                                  \
    {                                                   \
        struct logger* logger = get_default_logger();   \
        logger->log(logger, level, fmt, ##__VA_ARGS__); \
    } while (0)

#define TLOG_EMERG(fmt, ...)   LOG(LOG_EMERG, fmt, ##__VA_ARGS__)
#define TLOG_ALERT(fmt, ...)   LOG(LOG_ALERT, fmt, ##__VA_ARGS__)
#define TLOG_CRIT(fmt, ...)    LOG(LOG_CRIT, fmt, ##__VA_ARGS__)
#define TLOG_ERR(fmt, ...)     LOG(LOG_ERR, fmt, ##__VA_ARGS__)
#define TLOG_WARNING(fmt, ...) LOG(LOG_WARNING, fmt, ##__VA_ARGS__)
#define TLOG_NOTICE(fmt, ...)  LOG(LOG_NOTICE, fmt, ##__VA_ARGS__)
#define TLOG_INFO(fmt, ...)    LOG(LOG_INFO, fmt, ##__VA_ARGS__)
#define TLOG_DEBUG(fmt, ...)   LOG(LOG_DEBUG, fmt, ##__VA_ARGS__)

#define XLOG(level, fmt, ...)                 \
    LOG(level, "%s:%d ", __FILE__, __LINE__); \
    LOG(level, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif
