/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NVDLA_PRIV_CHECK_H
#define NVDLA_PRIV_CHECK_H

#include <iostream>
#include <sstream>

#include <stdint.h>

namespace nvdla
{

namespace priv
{

class Logger
{
public:
    enum Severity
    {
        INTERNAL_ERROR = 0,
        ERROR          = 1,
        WARNING        = 2,
        INFO           = 3
    };
    virtual void log(Severity severity, const char* msg);
    Logger();
    virtual ~Logger();
};

extern Logger* gLogger;

template<Logger::Severity S>
class LogStream : public std::ostream
{
    class Buf : public std::stringbuf
    {
    public:
        virtual int sync()
        {
            std::string s = str();
            while (!s.empty() && s[s.length() - 1] == '\n')
                s.erase(s.length() - 1);
            gLogger->log(S, s.c_str());
            str("");
            return 0;
        }
    };

    Buf buffer;
public:
    LogStream() : std::ostream(&buffer)
    {
    }
};

extern LogStream<Logger::INTERNAL_ERROR> gLogInternalError;
extern LogStream<Logger::ERROR> gLogError;
extern LogStream<Logger::WARNING> gLogWarning;
extern LogStream<Logger::INFO> gLogInfo;

} // nvdla::priv

} // nvdla


//
// "using" in headers is typically a bad idea.
// but here it's internal for a good cause.
//
using nvdla::priv::gLogInternalError;
using nvdla::priv::gLogError;
using nvdla::priv::gLogWarning;
using nvdla::priv::gLogInfo;


#ifdef _MSC_VER
#    define FN_NAME __FUNCTION__
#else
#    define FN_NAME __func__
#endif

#endif // NVDLA_PRIV_CHECK_H
