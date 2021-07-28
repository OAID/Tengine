/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef NVDLA_PRIV_TYPE_H
#define NVDLA_PRIV_TYPE_H

#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include <exception>

#include "nvdla/IType.h"
#include "priv/Check.h"

namespace nvdla
{

namespace priv
{

// Note: these are used as array indices.
enum IOD { IOD_Input = 0U, IOD_Output = 1U, IOD_Debug = 2U, IOD_Max = 3U };
enum IO  { IO_Input = 0U, IO_Output = 1U, IO_Max = 2U };

enum ELST { ELST_Elem = 0U, ELST_Line = 1U, ELST_Surf = 2U, ELST_Tensor = 3U, ELST_Max = 4U };

class NvErrorException
{
public:
    NvErrorException(NvDlaError e, const char *file, const char *function, size_t line) :
        m_e(e), m_file(file), m_function(function), m_line(line) { }

    NvDlaError m_e;
    const char *m_file;
    const char *m_function;
    size_t m_line;
};


//
// Wrapper for contiguous (starting at zero) sequence enumeration(s).
//
template <typename EnumClass, typename UnderlyingType = NvU8>
class SequenceEnum
{
public:
    typedef UnderlyingType underlying_type;

protected:
    underlying_type m_e;


    static const char * s_c_str;          // class name str
    static const char * const s_c_strs[]; // class enum strs
    static const size_t s_num_elements;

public:
    static const char * parameter_name_c_str() { return s_c_str; }
    const char *c_str() const { return s_c_strs[ m_e ]; }

    underlying_type v() const { return m_e; }
    EnumClass e() const      { return EnumClass(m_e); }
    bool valid() const { return m_e < s_num_elements; }
    static inline size_t num_elements() { return s_num_elements; }

    SequenceEnum &operator =(underlying_type rhs) {
        m_e = rhs;
        return *this;
    }

    bool operator ==(const SequenceEnum &rhs) const
    {
        return m_e == rhs.m_e;
    }


    SequenceEnum(EnumClass p) : m_e(p) { }
    SequenceEnum() : m_e( underlying_type(s_num_elements)) {/* note, invalid!*/ }
    SequenceEnum(underlying_type v) {
        if (v > s_num_elements) {
            v = s_num_elements;     // FIXME: reverse?
            // throw: out of range!?! (prefer)
            // or require check on validity?
            // or coerce to valid?
        }
        m_e = v;
    }

};

/*
template < typename EnumClass >
bool SequenceEnumCompare(const EnumClass& a, const EnumClass& b)
{
    return a.v() < b.v();
}
*/
template < typename EnumClass >
struct SequenceEnumCompare {
  bool operator() (const EnumClass& lhs, const EnumClass& rhs) const
  {return lhs.v() < rhs.v();}
};
//bool operator<(const EnumClass a, const EnumClass b)


} // nvdla::priv
} // nvdla


//
// note: preprocessor definitions must happen at global scope level per MISRA
//
#define GEN_ENUM(X, N) X = N,
#define GEN_STR(X, N)  #X,


//
// Note: it'd be possible to reduce the number of parameters here
// but the MISRA rules only allow a single ## or # per macro. :(
// e.g.: ENUM_PARAMETER_STATIC(X,Y) X##Parameter{} s_c_str = #X; ...
//
#define ENUM_PARAMETER_STATIC(X,Y,Z)                                    \
    template<> const char *const X::s_c_strs[] = { Y(GEN_STR) };        \
    template<> const char * X::s_c_str = Z;                             \
    template<> const size_t X::s_num_elements = sizeof(X::s_c_strs)/sizeof(X::s_c_strs[0]);

// note: use this from inside the 'priv' namespace as that's where the SequenceEnum template is specified
// the E parameter is priv implied but can be scoped further if needed.
#define SEQUENCE_ENUM_STATIC_MEMBERS(E, U, Y, Z)                        \
    template<> const char *const SequenceEnum<E, U>::s_c_strs[] = { Y(GEN_STR) }; \
    template<> const char *      SequenceEnum<E, U>::s_c_str = Z;       \
    template<> const size_t      SequenceEnum<E, U>::s_num_elements =   \
        sizeof(SequenceEnum<E, U>::s_c_strs) / sizeof(SequenceEnum<E, U>::s_c_strs[0]);



namespace nvdla
{
namespace priv
{

//
// this isn't meant to be complicated. just to reduce typing and errors.
//
template <typename L, typename R>
class BiMap
{
public:
    BiMap()  { }
    ~BiMap() { }

    typedef typename std::map<L, R>::iterator left_iterator;
    typedef typename std::map<R, L>::iterator right_iterator;

    //typedef typename std::map<L, R>::const_iterator left_const_iterator;
    //	typedef typename std::map<R, L>::const_iterator right_const_iterator;


    void insert(L l, R r) { m_left_right[l] = r; m_right_left[r] = l; }
    void remove(L l) { R r = m_left_right[l]; m_left_right.erase(l); m_right_left.erase(r); }

    left_iterator begin_left()   { return m_left_right.begin(); }
    left_iterator end_left()     { return m_left_right.end();   }
    left_iterator find_left(L l) { return m_left_right.find(l); }

    right_iterator begin_right()   { return m_right_left.begin(); }
    right_iterator end_right()     { return m_right_left.end();   }
    right_iterator find_right(R r) { return m_right_left.find(r); }

protected:
    std::map<L, R> m_left_right;
    std::map<R, L> m_right_left;
};

//
// PrivPair and PrivDiamond simplify management of the pointers necessary
// to track public interfaces, their private implementations and derivations
// of such which result in a diamond inheritance pattern.  These are simply
// fancy 2 and 4-tuples implemented by std::pair and 2x same.
// Note: with RTTI enabled this can all disappear as dynamic_cast<>()
// would be available instead ;(
//
template <typename I, typename P>
class PrivPair
{
public:
    // I 表示的是 接口 Interface
    // P 表示的就是 私密的 Private
    typedef I InterfaceType;
    typedef P PrivateType;

    PrivPair() : m_i_priv(0, 0) { }

    PrivPair(I i, P priv) :
        m_i_priv(i, priv)
    { }

    PrivPair(const PrivPair &p) :
        m_i_priv(p.m_i_priv)
    { }

    inline bool operator !() const { return (!m_i_priv.first) || (!m_i_priv.second); }
    inline bool operator ==(const PrivPair &rhs) const { return m_i_priv == rhs.m_i_priv; }
    inline bool operator <(const PrivPair &rhs) const { return m_i_priv < rhs.m_i_priv; }

    inline I i() const      { return m_i_priv.first;  }
    inline P priv() const   { return m_i_priv.second; }

protected:
    std::pair<I, P> m_i_priv;
};

template <typename BI, typename BP, typename DI, typename DP>
class PrivDiamond
{
public:
    typedef  BI BaseInterfaceType;
    typedef  BP BasePrivType;
    typedef  DI DerivedInterfaceType;
    typedef  DP DerivedPrivType;
    typedef PrivPair<BI *, BP *> BasePairType;
    typedef PrivPair<DI *, DP *> DerivedPairType;

    PrivDiamond() : m_base(0, 0), m_derived(0, 0) { }

    PrivDiamond(const PrivDiamond &d) :
        m_base(d.base()),
        m_derived(d.derived())
    { }

    PrivDiamond(const BasePairType &b, const DerivedPairType &d) :
        m_base(b),
        m_derived(d)
    { }

    PrivDiamond(BI *b_i, BP *b_priv, DI *d_i, DP *d_priv) :
        m_base(b_i, b_priv),
        m_derived(d_i, d_priv)
    { }

    inline bool operator !() const  { return  (!m_base) || (!m_derived); }

    inline BasePairType     base()    const  { return m_base;    }
    inline DerivedPairType  derived() const  { return m_derived; }

protected:
    BasePairType    m_base;
    DerivedPairType m_derived;
};


// simple converters, utils
static inline const std::string toString(int i)
{
    std::stringstream ss;
    ss << i;
    return ss.str();
}

template <typename I, typename O>
O saturate(const I& in) {
    return static_cast<O>(std::max(std::min(in, static_cast<I>(std::numeric_limits<O>::max())),
                          static_cast<I>(std::numeric_limits<O>::lowest())));
}

template <typename T> struct PtrPrintIdList
{
    PtrPrintIdList() : m_sep("") {}
    ~PtrPrintIdList() { }
    void operator()(T * & t)       { gLogInfo << m_sep << (t?t->id():std::string("null")); m_sep=std::string(", "); }
    void operator()(T * const & t) { gLogInfo << m_sep << (t?t->id():std::string("null")); m_sep=std::string(", "); }
    std::string m_sep;
};




} // nvdla::priv

} // nvdla

#endif // NVDLA_PRIV_TYPE_H
