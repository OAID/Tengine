#ifndef __COMPILER_HPP__
#define __COMPILER_HPP__

#include <string>

#define DLLEXPORT __attribute__((visibility("default")))

#ifdef __ANDROID__

namespace std {

template <typename T> std::string to_string(T);
}

#endif

#endif
