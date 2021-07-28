/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

#define ROUNDUP_AND_ALIGN(N, A)     (((N) % (A)) ? ((N) + (A) - ((N) % (A))) : (N))
#define ROUNDDOWN_AND_ALIGN(N, A)   (((N) % (A)) ? ((N) - ((N) % (A))) : (N))

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




#define ASSERT(cond) \
    if (!(cond)) { ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Condition Asserted"); }

#define API_CHECK(condition)											\
    do {																\
        if ((condition) == false) {										\
            std::ostringstream _error;									\
            gLogError << "Parameter check failed in " << FN_NAME;		\
            gLogError << ", file " << __FILE__ << " line " << __LINE__;	\
            gLogError << ", condition: " << #condition << std::endl;	\
                return;													\
        }																\
    } while(0)

#define API_CHECK_RETVAL(condition, retval)								\
    do {																\
        if ((condition) == false) {										\
            std::ostringstream _error;									\
            gLogError << "Parameter check failed in " << FN_NAME;		\
            gLogError << ", condition: " << #condition << std::endl;	\
                return retval;											\
        }																\
    } while(0)

#define API_CHECK_WEIGHTS(Name)	do {									\
        API_CHECK(Name.values != 0);                                    \
        API_CHECK(Name.count > 0);										\
        API_CHECK(DataType::UnderlyingType(Name.type) < EnumMax<DataType>()); \
    }	while(0)

#define API_CHECK_WEIGHTS0(Name) do {									\
        API_CHECK(Name.count >= 0);										\
        API_CHECK(Name.count > 0 ? (Name.values != 0) : (Name.values == 0)); \
        API_CHECK(DataType::UnderlyingType(Name.type) < EnumMax<DataType>()); \
    }while(0)

#define API_CHECK_WEIGHTS_RETVAL(Name, retval) do {						\
        API_CHECK_RETVAL(Name.values != 0, retval);                     \
        API_CHECK_RETVAL(Name.count > 0, retval);						\
        API_CHECK_RETVAL(DataType::UnderlyingType(Name.type) < EnumMax<DataType>(), retval); \
    } while(0)

#define API_CHECK_WEIGHTS0_RETVAL(Name, retval)	do {					\
        API_CHECK_RETVAL(Name.count >= 0, retval);						\
        API_CHECK_RETVAL(Name.count > 0 ? (Name.values != 0) : (Name.values == 0), retval); \
        API_CHECK_RETVAL(DataType::UnderlyingType(Name.type) < EnumMax<DataType>(), retval); \
    } while (0)


#define API_CHECK_NULL(param)                API_CHECK((param) != 0)
#define API_CHECK_NULL_RETVAL(param, retval) API_CHECK_RETVAL((param) != 0, retval)
#define API_CHECK_NULL_RET_NULL(ptr)         API_CHECK_NULL_RETVAL(ptr, 0)
#define API_CHECK_ENUM_RANGE(Type, val)      API_CHECK(int(val) >= 0 && int(val) < EnumMax<Type>())
#define API_CHECK_ENUM_RANGE_RETVAL(Type, val, retval) API_CHECK_RETVAL(int(val) >= 0 && int(val) < EnumMax<Type>(), retval)
#define API_CHECK_DIMS3_TENSOR_RETVAL(dims, retval)	API_CHECK_RETVAL(dims.c > 0 && dims.h > 0 && dims.w > 0 && uint64_t(dims.c)*uint64_t(dims.h)*uint64_t(dims.w) < MAX_TENSOR_SIZE, retval)
#define API_CHECK_DIMS4_TENSOR_RETVAL(dims, retval) API_CHECK_RETVAL(dims.n > 0 && dims.c > 0 && dims.h > 0 && dims.w > 0 && uint64_t(dims.c)*uint64_t(dims.h)*uint64_t(dims.w) < MAX_TENSOR_SIZE, retval)
#define API_CHECK_DIMS3_TENSOR(dims)  API_CHECK(dims.c > 0 && dims.h > 0 && dims.w > 0 && uint64_t(dims.c)*uint64_t(dims.h)*uint64_t(dims.w) < MAX_TENSOR_SIZE)



#define ACCESSOR_MUTATOR(Type, Name, CapName)       \
        Type CurrentScope :: get##CapName() const	\
        {											\
            return mParams.Name;					\
        }											\
                                                    \
        void CurrentScope :: set##CapName(Type val)	\
        {											\
            mParams.Name = val;						\
            /*mNetwork->markChanged(*this);*/		\
        }

#define ACCESSOR_MUTATOR_CHECK_ENUM(Type, Name, CapName)			\
    Type CurrentScope :: get##CapName() const						\
    {																\
        return mParams.Name;										\
    }																\
                                                                    \
    void CurrentScope :: set##CapName(Type Name)					\
    {																\
        API_CHECK(int(Name) >= 0 && int(Name) < EnumMax<Type>());	\
        mParams.Name = Name;										\
        /*mNetwork->markChanged(this);*/							\
    }

#define ACCESSOR_MUTATOR_CHECK_EXPR(Type, Name, CapName, CheckExpr)	\
    Type CurrentScope :: get##CapName() const						\
    {																\
        return mParams.Name;										\
    }																\
                                                                    \
    void CurrentScope :: set##CapName(Type Name)					\
    {																\
        API_CHECK(CheckExpr);										\
        mParams.Name = Name;										\
        /*		mNetwork->markChanged(this);*/						\
    }


#define ACCESSOR_MUTATOR_WEIGHTS(Type, Name, CapName)					\
    Weights CurrentScope :: get##CapName() const						\
    {																	\
        return mParams.Name;											\
    }																	\
                                                                        \
    void CurrentScope :: set##CapName(Weights Name)						\
    {																	\
        API_CHECK(Name.values != 0);									\
        API_CHECK(Name.count > 0);										\
        API_CHECK(DataType::UnderlyingType(Name.type) < EnumMax<DataType>()); \
        mParams.Name = Name;											\
        /*mNetwork->markChanged(this);*/								\
    }


//
// internal error checks (sends to gLogInternalError vs. gLogError)
//
#define CHECK_NULL(ptr)	do {                                \
        if (ptr == 0) {                                     \
            gLogInternalError << "error: input " <<	#ptr << \
                " is NULL in " << FN_NAME << std::endl;     \
                return;                                     \
        }                                                   \
    } while (0)

#define CHECK_NULL_RET_NULL(ptr) do {                       \
        if (ptr == 0) {                                     \
            gLogInternalError << "error: input " << #ptr << \
                " is NULL in " << FN_NAME << std::endl;     \
                return 0;                                   \
        }                                                   \
    } while(0)

#define CHECK_NULL_RET_VAL(ptr, val) do {                   \
        if (ptr == 0) {                                     \
            gLogInternalError << "error: input " << #ptr << \
                " is NULL in " << FN_NAME << std::endl;     \
                return val;                                 \
        }                                                   \
    } while(0)

#endif // NVDLA_PRIV_CHECK_H
