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

#include <stdio.h>
#include <time.h>

#include "sys_port.h"
#include "lock.h"
#include "tengine_c_api.h"
#include "tengine_log.h"

//#define DEFAULT_LOG_LEVEL LOG_INFO
#define DEFAULT_LOG_LEVEL LOG_DEBUG

static lock_t log_lock;
static const char* map_table[] = {"EMERG", "ALERT", "CRIT", "ERROR", "WARN", "NOTICE", "INFO", "DEBUG"};

static void do_log(struct logger* logger, int level, const char* fmt, ...)
{
    va_list ap;
    char msg[256];
    int max_len = 256;
    int left = max_len;
    char* p = msg;
    int ret;

    if (logger->log_level < level || level > LOG_DEBUG)
        return;
#ifndef CONFIG_ARCH_CORTEX_M
    if (logger->option.print_time)
    {
        time_t t = time(NULL);
        ret = strftime(p, left, "%Y-%m-%d %X ", localtime(&t));
        left -= ret;
        p += ret;
    }
#endif

    if (left <= 1)
        goto print;

    if (logger->option.print_level)
    {
        ret = snprintf(p, left, "%s ", map_table[level]);
        left -= ret;
        p += ret;
    }

    if (left <= 1)
        goto print;

    if (logger->option.print_prefix && logger->prefix)
    {
        ret = snprintf(p, left, "%s ", logger->prefix);
        left -= ret;
        p += ret;
    }

    if (left <= 1)
        goto print;

    va_start(ap, fmt);

    ret = vsnprintf(p, left, fmt, ap);

    va_end(ap);

print:
    msg[max_len - 1] = 0x0;

    lock(&log_lock);

    logger->output_func(msg);

    unlock(&log_lock);
}

static void change_log_level(struct logger* logger, int level)
{
    if (level < 0 || level > LOG_DEBUG)
        return;

    logger->log_level = level;
}

static void set_output_func(struct logger* logger, void (*func)(const char*))
{
    logger->output_func = func;
}

static void output_stderr(const char* msg)
{
    fprintf(stderr, "%s", msg);
}

struct logger* get_default_logger(void)
{
    static int inited = 0;
    static struct logger default_logger;

    if (inited)
        return &default_logger;
    else
        init_lock(&log_lock);

    lock(&log_lock);

    if (!inited)
    {
        inited = 1;

        default_logger.prefix = NULL;
        default_logger.log_level = DEFAULT_LOG_LEVEL;

        default_logger.output_func = output_stderr;
        default_logger.log = do_log;
        default_logger.set_log_level = change_log_level;
        default_logger.set_output_func = set_output_func;

        default_logger.option.print_prefix = 0;
        default_logger.option.print_time = 0;
        default_logger.option.print_level = 0;
    }

    unlock(&log_lock);

    return &default_logger;
}
