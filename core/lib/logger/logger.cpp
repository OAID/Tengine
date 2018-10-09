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
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "compiler.hpp"
#include "logger.hpp"

using namespace std::chrono;

namespace TEngine {

class StdioLogger : public Logger {
 public:
  StdioLogger() : log_stream_(std::cerr), null_stream_(std::ios_base::ate) {
    cur_log_level_ = kInfo;
  }

  bool SetLogLevel(LogLevel level) {
    cur_log_level_ = level;
    return true;
  }

  LogLevel GetLogLevel(void) { return cur_log_level_; }

  /*option part */

  bool SetLogOption(const LogOption& opt) {
    option_ = opt;
    return true;
  }

  LogOption GetLogOption(void) { return option_; }

  /*Log part */

  std::ostream& Log(LogLevel level);

  ~StdioLogger(){};

 private:
  LogLevel cur_log_level_;
  LogOption option_;
  std::ostream& log_stream_;
  std::ostringstream null_stream_;
};

std::ostream& StdioLogger::Log(LogLevel level) {
  if (level < cur_log_level_) {
    // clear dummy buffer
    null_stream_.str("");
    return null_stream_;
  }
#if 0
   if(option_.log_date)
   {
      auto t=system_clock::to_time_t(system_clock::now());
    
      log_stream_<<std::put_time(std::localtime(&t),"%Y-%m-%d %X ");

   }
#endif

  if (option_.log_level) {
    log_stream_ << std::string(LogLevelStr(level)) << " ";
  }

  if (option_.prefix.size()) {
    log_stream_ << option_.prefix;
  }

  return log_stream_;
}

static Logger* real_logger;

Logger* Logger::GetLogger(void) {
  static int have_been_called = 0;
  static StdioLogger default_logger;

  if (have_been_called) return real_logger;

  /* the real init part */

  have_been_called = 1;

  real_logger = &default_logger;

  LogOption option;

  option.log_date = false;
  option.log_level = false;

  real_logger->SetLogOption(option);

  return real_logger;
}

static const char* map_table[] = {"DEBUG", "INFO",  "WARN",
                                  "ERROR", "ALERT", "FATAL"};

const char* Logger::LogLevelStr(LogLevel level) { return map_table[level]; }

void Logger::SetLogger(Logger* new_logger) {
  Logger* logger = GetLogger();

  if (new_logger != logger) real_logger = new_logger;
}

}  // namespace TEngine
