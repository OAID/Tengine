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
#include <sstream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <mutex>
#include <ctime>

#include "compiler.hpp"
#include "logger.hpp"

using namespace std::chrono;

namespace TEngine {

class LogBuf : public std::basic_streambuf<char>
{
public:
    typedef void (*output_func_t)(const char*);

    LogBuf(int mem_size)
    {
        mem_size_ = mem_size;
        buf_size_ = mem_size - 2;
        disable_output_ = false;
        mem_ = ( char* )malloc(buf_size_);

        this->setp(mem_, mem_ + buf_size_);
    }

    void disable_output(void)
    {
        disable_output_ = true;
    }

    void enable_output(void)
    {
        disable_output_ = false;
    }

    static void set_output_func(output_func_t func)
    {
        output_func = func;
    }

    int overflow(int ch)
    {
        sync();

        this->sputc(ch);

        return ch;
    }

    int sync(void)
    {
        if(!disable_output_)
        {
            char* endptr = this->pptr();
            endptr[0] = 0x0;    // it is safe

            if(output_func == nullptr)
            {
                std::cerr << mem_;
            }
            else
            {
                shared_output(mem_);
            }
        }

        this->setp(mem_, mem_ + buf_size_);

        return 0;
    }

    ~LogBuf(void)
    {
        sync();
        free(mem_);
    }

    static output_func_t output_func;
    static void shared_output(const char* mem);

protected:
    char* mem_;
    int buf_size_;
    int mem_size_;
    bool disable_output_;
};

LogBuf::output_func_t LogBuf::output_func = nullptr;

void LogBuf::shared_output(const char* mem)
{
    static std::mutex output_lock;

    output_lock.lock();

    LogBuf::output_func(mem);

    output_lock.unlock();
}

class StdLogger : public Logger
{
public:
    StdLogger()
    {
        const char* log_level = std::getenv("TENGINE_LOG_LEVEL");

        if(log_level == nullptr)
            cur_log_level_ = kInfo;
        else
            cur_log_level_ = ( LogLevel )strtoul(log_level, NULL, 10);
    }

    bool SetLogLevel(LogLevel level)
    {
        cur_log_level_ = level;
        return true;
    }

    LogLevel GetLogLevel(void)
    {
        return cur_log_level_;
    }

    void SetLogOutputFunc(void (*func)(const char*))
    {
        LogBuf::set_output_func(func);
    }

    /*option part */

    bool SetLogOption(const LogOption& opt)
    {
        option_ = opt;
        return true;
    }

    LogOption GetLogOption(void)
    {
        return option_;
    }

    /*Log part */

    log_stream_t Log(LogLevel level);

    ~StdLogger(){};

private:
    LogLevel cur_log_level_;
    LogOption option_;
};

struct log_ostream : public std::ostream
{
    log_ostream(LogBuf* buf) : std::ostream(buf)
    {
        log_buf_ = buf;
    }

    void disable_output(void)
    {
        log_buf_->disable_output();
    }

    void enable_output(void)
    {
        log_buf_->enable_output();
    }

    ~log_ostream(void)
    {
        delete log_buf_;
    }

    LogBuf* log_buf_;
};

log_stream_t StdLogger::Log(LogLevel level)
{
    LogBuf* log_buf = new LogBuf(128);

    log_ostream* log_stream = new log_ostream(log_buf);

    if(level > cur_log_level_)
        log_stream->disable_output();
    else
        log_stream->enable_output();

    if(option_.log_date)
    {
        auto t = system_clock::to_time_t(system_clock::now());
#if defined(__GNUC__) && __GNUC__ > 5
        (*log_stream) << std::put_time(std::localtime(&t), "%Y-%m-%d %X ");
#else
        char buf[128];
        strftime(buf, 128, "%Y-%m-%d %X ", localtime(&t));
        (*log_stream) << buf;
#endif
    }

    if(option_.log_level)
    {
        (*log_stream) << std::string(LogLevelStr(level)) << " ";
    }

    if(option_.prefix.size())
    {
        (*log_stream) << option_.prefix << " ";
    }

    return log_stream_t(log_stream);
}

static Logger* real_logger;

Logger* Logger::GetLogger(void)
{
    static int have_been_called = 0;
    static StdLogger default_logger;

    if(have_been_called)
        return real_logger;

    /* the real init part */

    have_been_called = 1;

    real_logger = &default_logger;

    LogOption option;

    const char* env_str = std::getenv("TENGINE_LOG_DATE");

    if(env_str && env_str[0] == '1')
        option.log_date = true;
    else
        option.log_date = false;

    env_str = getenv("TENGINE_LOG_EVENTLEVEL");

    if(env_str && env_str[0] == '1')
        option.log_level = true;
    else
        option.log_level = false;

    env_str = getenv("TENGINE_LOG_PREFIX");

    if(env_str)
        option.prefix = env_str;

    real_logger->SetLogOption(option);

    return real_logger;
}

static const char* map_table[] = {"EMERG", "ALERT", "CRIT", "ERROR", "WARN", "NOTICE", "INFO", "DEBUG"};

const char* Logger::LogLevelStr(LogLevel level)
{
    return map_table[level];
}

void Logger::SetLogger(Logger* new_logger)
{
    Logger* logger = GetLogger();

    if(new_logger != logger)
        real_logger = new_logger;
}

}    // namespace TEngine
