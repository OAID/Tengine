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

#ifndef NVDLA_UTILS_ERROR_MACROS_H
#define NVDLA_UTILS_ERROR_MACROS_H

#include <stdint.h>
#include <stdbool.h>
#include <dlaerror.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define NVDLA_UNUSED(expr) do { (void)(expr); } while (0)

const char* NvDlaUtilsGetNvErrorString(NvDlaError e);

/**
 * There are two ways for a client to customize the error logging macros, both of which
 * are performed by defining a macro value from the client before including this file:
 *  1) Define NVDLA_UTILS_ERROR_TAG to be a string with the client name. This string will
 *     be appended to any error logs before being output by the default NvDlaUtilsLogError.
 *     In addition, NVDLA_UTILS_ERROR_PATH may be optionally defined so that this path will
 *     be stripped from any generated errors (if not set, the 'nvidia/tegra/cv/dla' path
 *     will be stripped by default).
 *  2) Define NVDLA_UTILS_LOG_ERROR to point to a custom error logging function within the
 *     client. The signature for this macro should be the same as the default, below.
 */
#if defined (NVDLA_UTILS_ERROR_TAG)
// Use the default NvDlaUtilsLogError with the client tag appended.
void NvDlaUtilsLogError(const char* tag, const char* path, NvDlaError e, const char* file, const char* func,
                        uint32_t line, bool propagating, const char* format, ...);
#if !defined (NVDLA_UTILS_ERROR_PATH)
#define NVDLA_UTILS_ERROR_PATH "dla"
#endif

#define NVDLA_UTILS_LOG_ERROR(_err, _file, _func, _line, _propagating, _format, ...) \
    do { \
        NvDlaUtilsLogError(NVDLA_UTILS_ERROR_TAG, NVDLA_UTILS_ERROR_PATH, \
                                (_err), (_file), (_func), (_line), \
                                (_propagating), (_format), ##__VA_ARGS__); \
    } \
    while (0)
#elif defined (NVDLA_UTILS_LOG_ERROR)
// Use the client's custom error logging function.
#else
#error "One of NVDLA_UTILS_ERROR_TAG or NVDLA_UTILS_LOG_ERROR must be defined"
#endif

/**
 * Argument counting macros.
 */
#define NVDLA_UTILS_CAT(A, B) A##B
#define NVDLA_UTILS_SELECT(NAME, NUM) NVDLA_UTILS_CAT(NAME##_, NUM)
#define NVDLA_UTILS_GET_COUNT(_1, _2, _3, _4, _5, COUNT, ...) COUNT
#define NVDLA_UTILS_VA_SIZE(...) NVDLA_UTILS_GET_COUNT(__VA_ARGS__, 5, 4, 3, 2, 1)
#define NVDLA_UTILS_VA_SELECT(NAME, ...) NVDLA_UTILS_SELECT(NAME, NVDLA_UTILS_VA_SIZE(__VA_ARGS__))(__VA_ARGS__)

/**
 * Simply report an error.
 */
#define REPORT_ERROR(...) NVDLA_UTILS_VA_SELECT(REPORT_ERROR_IMPL, __VA_ARGS__)
#define REPORT_ERROR_IMPL_1(_err) \
    do { NVDLA_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, 0); } while (0)
#define REPORT_ERROR_IMPL_2(_err, _format) \
    do { NVDLA_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format)); } while (0)
#define REPORT_ERROR_IMPL_3(_err, _format, ...) REPORT_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_IMPL_4(_err, _format, ...) REPORT_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_IMPL_5(_err, _format, ...) REPORT_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define REPORT_ERROR_IMPL_N(_err, _format, ...) \
    do { NVDLA_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format), __VA_ARGS__); } while (0)

/**
 * Report and return an error that was first detected in the current method.
 */
#define ORIGINATE_ERROR(...) NVDLA_UTILS_VA_SELECT(ORIGINATE_ERROR_IMPL, __VA_ARGS__)
#define ORIGINATE_ERROR_IMPL_1(_err) \
    do { \
        NVDLA_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, 0); \
        return (_err); \
    } while (0)
#define ORIGINATE_ERROR_IMPL_2(_err, _format) \
    do { \
        NVDLA_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format)); \
        return (_err); \
    } while (0)
#define ORIGINATE_ERROR_IMPL_3(_err, _format, ...) ORIGINATE_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define ORIGINATE_ERROR_IMPL_4(_err, _format, ...) ORIGINATE_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define ORIGINATE_ERROR_IMPL_5(_err, _format, ...) ORIGINATE_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define ORIGINATE_ERROR_IMPL_N(_err, _format, ...) \
    do { \
        NVDLA_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format), __VA_ARGS__); \
        return (_err); \
    } while (0)

/**
 * Report an error that was first detected in the current method, then jumps to the "fail:" label.
 * The variable "NvError e" must have been previously declared.
 */
#define ORIGINATE_ERROR_FAIL(...) NVDLA_UTILS_VA_SELECT(ORIGINATE_ERROR_FAIL_IMPL, __VA_ARGS__)
#define ORIGINATE_ERROR_FAIL_IMPL_1(_err) \
    do { \
        e = (_err); \
        NVDLA_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, 0); \
        goto fail; \
    } while (0)
#define ORIGINATE_ERROR_FAIL_IMPL_2(_err, _format) \
    do { \
        e = (_err); \
        NVDLA_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format)); \
        goto fail; \
    } while (0)
#define ORIGINATE_ERROR_FAIL_IMPL_3(_err, _format, ...) ORIGINATE_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define ORIGINATE_ERROR_FAIL_IMPL_4(_err, _format, ...) ORIGINATE_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define ORIGINATE_ERROR_FAIL_IMPL_5(_err, _format, ...) ORIGINATE_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define ORIGINATE_ERROR_FAIL_IMPL_N(_err, _format, ...) \
    do { \
        e = (_err); \
        NVDLA_UTILS_LOG_ERROR((_err), __FILE__, __FUNCTION__, __LINE__, false, (_format), __VA_ARGS__); \
        goto fail; \
    } while (0)

/**
 * Calls another function, and if an error was returned it is reported and returned.
 */
#define PROPAGATE_ERROR(...) NVDLA_UTILS_VA_SELECT(PROPAGATE_ERROR_IMPL, __VA_ARGS__)
#define PROPAGATE_ERROR_IMPL_1(_err) \
    do { \
        NvDlaError peResult = (_err); \
        if (peResult != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, 0); \
            return peResult; \
        } \
    } while (0)
#define PROPAGATE_ERROR_IMPL_2(_err, _format) \
    do { \
        NvDlaError peResult = (_err); \
        if (peResult != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, (_format)); \
            return peResult; \
        } \
    } while (0)
#define PROPAGATE_ERROR_IMPL_3(_err, _format, ...) PROPAGATE_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_IMPL_4(_err, _format, ...) PROPAGATE_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_IMPL_5(_err, _format, ...) PROPAGATE_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_IMPL_N(_err, _format, ...) \
    do { \
        NvDlaError peResult = (_err); \
        if (peResult != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, (_format), __VA_ARGS__); \
            return peResult; \
        } \
    } while (0)

/**
 * Calls another function, and if an error was returned it is reported before jumping to the
 * "fail:" label. The variable "NvError e" must have been previously declared.
 */
#define PROPAGATE_ERROR_FAIL(...) NVDLA_UTILS_VA_SELECT(PROPAGATE_ERROR_FAIL_IMPL, __VA_ARGS__)
#define PROPAGATE_ERROR_FAIL_IMPL_1(_err) \
    do { \
        e = (_err); \
        if (e != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, 0); \
            goto fail; \
        } \
    } while (0)
#define PROPAGATE_ERROR_FAIL_IMPL_2(_err, _format) \
    do { \
        e = (_err); \
        if (e != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format)); \
            goto fail; \
        } \
    } while (0)
#define PROPAGATE_ERROR_FAIL_IMPL_3(_err, _format, ...) PROPAGATE_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_FAIL_IMPL_4(_err, _format, ...) PROPAGATE_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_FAIL_IMPL_5(_err, _format, ...) PROPAGATE_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_FAIL_IMPL_N(_err, _format, ...) \
    do { \
        e = (_err); \
        if (e != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format), __VA_ARGS__); \
            goto fail; \
        } \
    } while (0)

/**
 * Calls another function, and if an error was returned it is reported and assigned to the
 * variable 'e' (which must have been previously declared). The caller does not return.
 */
#define PROPAGATE_ERROR_CONTINUE(...) NVDLA_UTILS_VA_SELECT(PROPAGATE_ERROR_CONTINUE_IMPL, __VA_ARGS__)
#define PROPAGATE_ERROR_CONTINUE_IMPL_1(_err) \
    do { \
        NvDlaError peResult = (_err); \
        if (peResult != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, 0); \
            if (e == NvDlaSuccess) \
                e = peResult; \
        } \
    } while (0)
#define PROPAGATE_ERROR_CONTINUE_IMPL_2(_err, _format) \
    do { \
        NvDlaError peResult = (_err); \
        if (peResult != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, (_format)); \
            if (e == NvDlaSuccess) \
                e = peResult; \
        } \
    } while (0)
#define PROPAGATE_ERROR_CONTINUE_IMPL_3(_err, _format, ...) PROPAGATE_ERROR_CONTINUE_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_CONTINUE_IMPL_4(_err, _format, ...) PROPAGATE_ERROR_CONTINUE_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_CONTINUE_IMPL_5(_err, _format, ...) PROPAGATE_ERROR_CONTINUE_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_CONTINUE_IMPL_N(_err, _format, ...) \
    do { \
        NvDlaError peResult = (_err); \
        if (peResult != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(peResult, __FILE__, __FUNCTION__, __LINE__, true, (_format), __VA_ARGS__); \
            if (e == NvDlaSuccess) \
                e = peResult; \
        } \
    } while (0)




#ifdef __cplusplus

#define THROW_ERROR(...) NVDLA_UTILS_VA_SELECT(THROW_ERROR_IMPL, __VA_ARGS__)
#define THROW_ERROR_IMPL_1(_err)                                        \
    do {                                                                \
        e = (_err);                                                     \
        NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, 0); \
        throw(nvdla::priv::NvErrorException(e, __FILE__, __FUNCTION__, __LINE__)); \
    } while (0)

#define THROW_ERROR_IMPL_2(_err, _format)                               \
    do {                                                                \
        e = (_err);                                                     \
        NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format)); \
        throw(nvdla::priv::NvErrorException(e, __FILE__, __FUNCTION__, __LINE__)); \
    } while (0)
#define THROW_ERROR_IMPL_3(_err, _format, ...) THROW_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define THROW_ERROR_IMPL_4(_err, _format, ...) THROW_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define THROW_ERROR_IMPL_5(_err, _format, ...) THROW_ERROR_IMPL_N((_err), (_format), __VA_ARGS__)
#define THROW_ERROR_IMPL_N(_err, _format, ...)                          \
    do {                                                                \
        e = (_err);                                                     \
        NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format), __VA_ARGS__); \
        throw(nvdla::priv::NvErrorException(e, __FILE__, __FUNCTION__, __LINE__)); \
    } while (0)


/**
 * Calls another function, and if an error was returned it is reported before throwing
 * an NvErrorException. The variable "NvError e" must have been previously declared.
 */
#define PROPAGATE_ERROR_THROW(...) NVDLA_UTILS_VA_SELECT(PROPAGATE_ERROR_THROW_IMPL, __VA_ARGS__)
#define PROPAGATE_ERROR_THROW_IMPL_1(_err) \
    do { \
        e = (_err); \
        if (e != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, 0); \
            throw(nvdla::priv::NvErrorException(e, __FILE__, __FUNCTION__, __LINE__)); \
        } \
    } while (0)


#define PROPAGATE_ERROR_THROW_IMPL_2(_err, _format) \
    do { \
        e = (_err); \
        if (e != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format)); \
            throw(nvdla::priv::NvErrorException(e, __FILE__, __FUNCTION__, __LINE__)); \
            goto fail; \
        } \
    } while (0)
#define PROPAGATE_ERROR_THROW_IMPL_3(_err, _format, ...) PROPAGATE_ERROR_THROW_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_THROW_IMPL_4(_err, _format, ...) PROPAGATE_ERROR_THROW_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_THROW_IMPL_5(_err, _format, ...) PROPAGATE_ERROR_THROW_IMPL_N((_err), (_format), __VA_ARGS__)
#define PROPAGATE_ERROR_THROW_IMPL_N(_err, _format, ...) \
    do { \
        e = (_err); \
        if (e != NvDlaSuccess) \
        { \
            NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format), __VA_ARGS__); \
            throw(nvdla::priv::NvErrorException(e, __FILE__, __FUNCTION__, __LINE__)); \
            goto fail; \
        } \
    } while (0)


/**
 * Calls another function, and if an error was returned or caught it is reported before jumping to the
 * "fail:" label. The variable "NvError e" must have been previously declared.
 */
#define CATCH_PROPAGATE_ERROR_FAIL(...) NVDLA_UTILS_VA_SELECT(CATCH_PROPAGATE_ERROR_FAIL_IMPL, __VA_ARGS__)
#define CATCH_PROPAGATE_ERROR_FAIL_IMPL_1(_err) \
    try                                         \
    {                                           \
        e = (_err);                             \
        if (e != NvDlaSuccess)                     \
        {                                                               \
            NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, 0); \
            goto fail;                                                  \
        }                                                               \
    }                                                                   \
    catch (const NvErrorException & ee)                                 \
    {                                                                   \
        NVDLA_UTILS_LOG_ERROR(ee.m_e, ee.m_file, ee.m_function, ee.m_line, true, 0); \
        e = ee.m_e;                                                     \
        goto fail;                                                      \
    }

#define CATCH_PROPAGATE_ERROR_FAIL_IMPL_2(_err, _format)    \
    try                                                     \
    {                                                       \
        e = (_err);                                         \
        if (e != NvDlaSuccess)                                 \
        {                                                               \
            NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format)); \
            goto fail;                                                  \
        }                                                               \
    }                                                                   \
    catch (const NvErrorException & ee)                                 \
    {                                                                   \
        NVDLA_UTILS_LOG_ERROR(ee.m_e, ee.m_file, ee.m_function, true, (_format)); \
        e = ee.m_e;                                                     \
        goto fail;                                                      \
    }

#define CATCH_PROPAGATE_ERROR_FAIL_IMPL_3(_err, _format, ...) CATCH_PROPAGATE_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define CATCH_PROPAGATE_ERROR_FAIL_IMPL_4(_err, _format, ...) CATCH_PROPAGATE_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define CATCH_PROPAGATE_ERROR_FAIL_IMPL_5(_err, _format, ...) CATCH_PROPAGATE_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define CATCH_PROPAGATE_ERROR_FAIL_IMPL_N(_err, _format, ...) \
    try                                                       \
    {                                                         \
        e = (_err);                                           \
        if (e != NvDlaSuccess)                                   \
        {                                                               \
            NVDLA_UTILS_LOG_ERROR(e, __FILE__, __FUNCTION__, __LINE__, true, (_format), __VA_ARGS__); \
            goto fail;                                                  \
        }                                                               \
    }                                                                   \
    catch (const NvErrorException & ee)                                 \
    {                                                                   \
        NVDLA_UTILS_LOG_ERROR(ee.m_e, ee.m_file, ee.m_function, ee.m_line, true, (_format), __VA_ARGS__); \
        e = ee.m_e;                                                     \
        goto fail;                                                      \
    }



/**
 * Calls another function, and if an error was caught it is reported before jumping to the
 * "fail:" label.
 */
#define CATCH_ERROR_FAIL(...) NVDLA_UTILS_VA_SELECT(CATCH_ERROR_FAIL_IMPL, __VA_ARGS__)
#define CATCH_ERROR_FAIL_IMPL_1(_err) \
    try                                         \
    {                                           \
        _err;                             \
    }                                                                   \
    catch (const NvErrorException & ee)                                 \
    {                                                                   \
        NVDLA_UTILS_LOG_ERROR(ee.m_e, ee.m_file, ee.m_function, ee.m_line, true, 0); \
        e = ee.m_e;                                                     \
        goto fail;                                                      \
    }

#define CATCH_ERROR_FAIL_IMPL_2(_err, _format)    \
    try                                                     \
    {                                                       \
        _err;                                         \
    }                                                                   \
    catch (const NvErrorException & ee)                                 \
    {                                                                   \
        NVDLA_UTILS_LOG_ERROR(ee.m_e, ee.m_file, ee.m_function, true, (_format)); \
        e = ee.m_e;                                                     \
        goto fail;                                                      \
    }

#define CATCH_ERROR_FAIL_IMPL_3(_err, _format, ...) CATCH_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define CATCH_ERROR_FAIL_IMPL_4(_err, _format, ...) CATCH_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define CATCH_ERROR_FAIL_IMPL_5(_err, _format, ...) CATCH_ERROR_FAIL_IMPL_N((_err), (_format), __VA_ARGS__)
#define CATCH_ERROR_FAIL_IMPL_N(_err, _format, ...) \
    try                                                       \
    {                                                         \
        _err;                                           \
    }                                                                   \
    catch (const NvErrorException & ee)                                 \
    {                                                                   \
        NVDLA_UTILS_LOG_ERROR(ee.m_e, ee.m_file, ee.m_function, ee.m_line, true, (_format), __VA_ARGS__); \
        e = ee.m_e;                                                     \
        goto fail;                                                      \
    }


#endif // __cplusplus


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // NVDLA_UTILS_ERROR_MACROS_H
