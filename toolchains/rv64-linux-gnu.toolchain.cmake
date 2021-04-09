set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR rv64)

set(CMAKE_ASM_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_C_COMPILER "riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "riscv64-unknown-linux-gnu-g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-march=rv64imafdcvxtheadc -mabi=lp64dv -mtune=c910 -mfp16 -lc")
set(CMAKE_CXX_FLAGS "-march=rv64imafdcvxtheadc -mabi=lp64dv -mtune=c910 -mfp16 -lc")

# cache flags for C910v
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")