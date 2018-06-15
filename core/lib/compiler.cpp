#include <cstdio>
#include <cstdlib>
#include <string>

#include "compiler.hpp"

#ifdef ANDROID

namespace std {
template <>
std::string to_string<int> (int n)
{
    char buf[128];
    snprintf(buf,127,"%d",n);
    buf[127]=0x0;

    return std::string(buf);
}
}

#endif
