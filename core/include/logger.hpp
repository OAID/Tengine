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
#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include <string>


/* this file defines the interface of each logger implementation */
namespace TEngine {

enum LogLevel
{
  kDebug,
  kInfo,
  kWarn,
  kError,
  kAlert,
  kFatal
};


struct LogOption 
{
   std::string prefix;
   int  max_line_size;
   int  rate_limit;  /* KB per second */
   bool log_level;
   bool log_date;
};

struct  Logger {

    virtual bool SetLogLevel(LogLevel level)=0;
    virtual LogLevel GetLogLevel(void)=0;
    
    /*option part */

    virtual bool SetLogOption(const LogOption& opt)=0;

    virtual LogOption GetLogOption(void)=0;

    /*Log part */

    virtual std::ostream& Log(LogLevel)=0;
    
    virtual ~Logger(){};

    static Logger * GetLogger(); /* get the global logger */
    static void  SetLogger(Logger * log); /* set the global logger */
    static const char* LogLevelStr(LogLevel level);

};


/* the wrapper for other to use log utilities */

#define DO_LOG(level) 		Logger::GetLogger()->Log(level)
#define GET_LOG_OPTION()	Logger::GetLogger()->GetLogOption()
#define SET_LOG_OPTION(opt) 	Logger::GetLogger()->SetLogOption(opt)
#define GET_LOG_LEVEL()         Logger::GetLogger()->GetLogLevel()
#define SET_LOG_LEVEL(l)        Logger::GetLogger()->SetLogLevel(l)


#define LOG_DEBUG()  		DO_LOG(kDebug)
#define LOG_INFO()   	        DO_LOG(kInfo)
#define LOG_WARN()   	        DO_LOG(kWarn)
#define LOG_ERROR()  		DO_LOG(kError)
#define LOG_ALERT()  		DO_LOG(kAlert)
#define LOG_FATAL()  		DO_LOG(kFatal)


#define XLOG_DEBUG()  		LOG_DEBUG()<<__FILE___<<":"<<__LINE__<<" "
#define XLOG_INFO()   	        LOG_INFO()<<__FILE__<<":"<<__LINE__<<" "
#define XLOG_WARN()   	        LOG_WARN()<<__FILE__<<":"<<__LINE__<<" "
#define XLOG_ERROR()  		LOG_ERROR()<<__FILE__<<":"<<__LINE__<<" "
#define XLOG_ALERT()  		LOG_ALERT()<<__FILE__<<":"<<__LINE__<<" "
#define XLOG_FATAL()  		LOG_FATAL()<<__FILE__<<":"<<__LINE__<<" "


} //namespace TEngine


#endif
