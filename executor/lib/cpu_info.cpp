
#include "cpu_info.hpp"
#include "cpu_device.h"

namespace TEngine {

/* MUST be the same order in cpu_device.h */

static const char* cpu_arch_table[] = {
    "generic",
    "arm64",
    "arm32",
    "armv8.2",
};

static const char* cpu_model_table[] = {
    "generic", "A72", "A53", "A17", "A7", "A55", "Kyro",
};

const char* CPUInfo::GetCPUArchString(int cpu_id) const {
  int cpu_arch = GetCPUArch(cpu_id);

  if (cpu_arch < 0) return nullptr;

  return cpu_arch_table[cpu_arch];
}

const char* CPUInfo::GetCPUModelString(int cpu_id) const {
  int cpu_model = GetCPUModel(cpu_id);

  if (cpu_model < 0) return nullptr;

  return cpu_model_table[cpu_model];
}

}  // namespace TEngine
