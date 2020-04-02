#ifndef __COMPILER_HPP__
#define __COMPILER_HPP__

#include <string>

<<<<<<< HEAD
=======
#define DLLEXPORT __attribute__((visibility("default")))

>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
#ifdef __ANDROID__

namespace std {

template <typename T> std::string to_string(T);
}

#endif

#endif
