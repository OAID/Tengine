# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET (CMAKE_SYSTEM_NAME Linux)
SET (CMAKE_SYSTEM_PROCESSOR mips64r2)

# when mips-linux-gnu-gcc was installed, toolchain was available as below:
SET (CMAKE_C_COMPILER   "mips-linux-gnu-gcc")
SET (CMAKE_CXX_COMPILER "mips-linux-gnu-g++")

# set searching rules for cross-compiler
SET (CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET (CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# other needed options
SET (TENGINE_TOOLCHAIN_FLAG -march=mips64r2 -mabi=64 -mmsa -mhard-float -mfp64)

# do not skip OpenMP check as default
SET (TENGINE_FORCE_SKIP_OPENMP OFF)
