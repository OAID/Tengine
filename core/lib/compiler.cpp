#include <cstdio>
#include <cstdlib>
#include <string>

#include "compiler.hpp"

#ifdef ANDROID

namespace std {
template <>
std::string to_string<int>(int n) {
  char buf[128];
  snprintf(buf, 127, "%d", n);
  buf[127] = 0x0;

  return std::string(buf);
}
}  // namespace std

#endif

#ifdef STATIC_BUILD

extern "C" void __pthread_cond_broadcast(void);
extern "C" void __pthread_cond_destroy(void);
extern "C" void __pthread_cond_signal(void);
extern "C" void __pthread_cond_wait(void);

void static_compiling_workaround(void) {
  __pthread_cond_broadcast();
  __pthread_cond_destroy();
  __pthread_cond_signal();
  __pthread_cond_wait();
}

#endif
