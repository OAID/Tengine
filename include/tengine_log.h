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

#ifndef __TENGINE_LOG_H__
#define __TENGINE_LOG_H__

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

    void (*log)(struct logger*, int level, const char* fmt, ...);
    void (*set_log_level)(struct logger*, int level);
    void (*set_output_func)(struct logger*, void (*func)(const char*));
};

extern struct logger* get_default_logger(void);

#define SET_LOG_OUTPUT(func)                          \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->set_output_func(logger, func);        \
    } while(0)

#define SET_LOG_LEVEL(level)                          \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->set_log_level(logger, level);         \
    } while(0)

#define SET_LOG_PRINT_TIME(val)                       \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->option.print_time = val;              \
    } while(0)

#define SET_LOG_PRINT_LEVEL(val)                      \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->option.print_level = val;             \
    } while(0)

#define SET_LOG_PRINT_PREFIX(val)                     \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->option.print_prefix = val;            \
    } while(0)

#define SET_LOG_PREFIX(prefix)                        \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->prefix = prefix;                      \
    } while(0)

#define LOG(level, fmt, content...)                   \
    do                                                \
    {                                                 \
        struct logger* logger = get_default_logger(); \
        logger->log(logger, level, fmt, ##content);   \
    } while(0)

#define TLOG_EMERG(fmt, content...) LOG(LOG_EMERG, fmt, ##content)
#define TLOG_ALERT(fmt, content...) LOG(LOG_ALERT, fmt, ##content)
#define TLOG_CRIT(fmt, content...) LOG(LOG_CRIT, fmt, ##content)
#define TLOG_ERR(fmt, content...) LOG(LOG_ERR, fmt, ##content)
#define TLOG_WARNING(fmt, content...) LOG(LOG_WARNING, fmt, ##content)
#define TLOG_NOTICE(fmt, content...) LOG(LOG_NOTICE, fmt, ##content)
#define TLOG_INFO(fmt, content...) LOG(LOG_INFO, fmt, ##content)
#define TLOG_DEBUG(fmt, content...) LOG(LOG_DEBUG, fmt, ##content)

#define XLOG(level, fmt, content...)          \
    LOG(level, "%s:%d ", __FILE__, __LINE__); \
    LOG(level, fmt, ##content)

#ifdef __cplusplus
}
#endif

#endif
